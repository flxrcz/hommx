"""Small helper functions used throughout the library

functions:
    mesh_from_file: Reads a mesh from a file to be used in dolfinx.
    rescale_mesh: copies and rescales/shifts a mesh
    rescale_mesh_in_place: rescales/shifts a mesh in place.
    create_periodic_boundary_conditions: creates periodic boundary conditions on a 2D box mesh.
"""

from collections.abc import Callable

import basix.ufl
import dolfinx
import numpy as np
import pyvista
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx_mpc import MultiPointConstraint
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner


def solve_diffusion_1d(
    epsilon: float, nx: int, A_callable: callable
) -> np.ndarray[tuple[int], np.dtype[float]]:
    r"""Solves diffusion equation with multiscale coefficient.
    Solves the diffusion equation:

    $$
    \mathrm{div}(A \\nabla u) = f
    $$

    Where $A$ is of the shape:

    $$
    A(x) = 1 + 0.5*\\sin(x\\frac{2*\pi}{\\varepsilon})
    $$

    Args:
        epsilon: $\\varepsilon$
        nx: discretization of the FEM solution
        A_callable: callable, such that A_callable(x) returns a proper ufl.Form where x are the Spatial Coordinates
    """
    msh = mesh.create_interval(comm=MPI.COMM_WORLD, nx=nx, points=(0.0, 1.0))
    V = fem.functionspace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=msh.topology.dim - 1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 1
    A = A_callable(x)
    a = inner(A * grad(u), grad(v)) * dx
    L = inner(f, v) * dx  # + inner(g, v) * ds # only dirichlet
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh.x.array


def solve_diffusion_2d(
    epsilon: float, nx: int, A_callable: callable
) -> np.ndarray[tuple[int], np.dtype[float]]:
    r"""Solves diffusion equation with multiscale coefficient.
    Solves the diffusion equation:

    $$
    \mathrm{div}(A \\nabla u) = f
    $$

    Where $A$ is of the shape:

    $$
    A(x) = 1 + 0.5*\\sin(x\\frac{2*\pi}{\\varepsilon})
    $$

    Args:
        epsilon: $\\varepsilon$
        nx: discretization of the FEM solution
        A_callable: callable, such that A_callable(x) returns a proper ufl.Form where x are the Spatial Coordinates
    """
    msh = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=nx, ny=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0),
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=msh.topology.dim - 1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 1
    A = A_callable(x)
    a = inner(A * grad(u), grad(v)) * dx
    L = inner(f, v) * dx  # + inner(g, v) * ds # only dirichlet
    problem = LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show()
    coords = V.tabulate_dof_coordinates()
    ix = np.argsort(coords[:, 1] + coords[:, 0] * 1e3)
    sorted_values = uh.x.array[ix]
    return sorted_values.reshape(nx + 1, nx + 1)


def mesh_from_file(filename):
    data = np.load(filename)
    points = data["points"]
    elements = data["elements"]
    return mesh_from_delaunay(points, elements)


def mesh_from_delaunay(
    points: np.ndarray[tuple[int, int], np.dtype[float]],
    triangles: np.ndarray[tuple[int, int], np.dtype[int]],
) -> mesh.Mesh:
    """Creates mesh from list of points and triangles.

    Args:
        points: (N_points, 2) shape array points on the plane.
        triangles: (N_triangles, 3) shape array containing indices into the points array
            where each row represents one triangle.

    Notes:
        Setting up the unit square could look something like this:
        ```py
        from scipy.spatial import Delaunay
        import numpy as np

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        points - np.stack([X, Y], axis=-1).reshape(-1, 2)
        triangles = Delaunay(points).simplices
        ```
    """
    basix_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, triangles, points, basix_element)
    return msh


def rescale_mesh(
    msh: mesh.Mesh,
    scale: float = 1,
    shift: np.ndarray[tuple[int], np.dtype[float]] = np.array([0, 0, 0]),
) -> mesh.Mesh:
    r"""Creates a rescaled and shifted copy of the mesh.

    Args:
        msh: Mesh
        scale: constant scalign factor
        shift: (3,) shape array containing the shift.

    For example consider the unit-square $[0,1]^2$.
    This is remapped to $\text{scale}*[0, 1]^2 + \text{shift} = [\text{shift}[0], \text{scale}+\text{shift}[1]]^2$.
    This is used to map between the Unit cell $Y=[0,1]^d$ and $Y_\varepsilon$ in the HMM.
    """
    shift = np.asarray(shift)
    assert len(shift.shape) == 1, "Only constant shift allowed, please supply shift with shape (3,)"
    assert shift.shape[0] == 3, "Need 3D shift, i.e. shift.shape=(3,)"
    points = msh.geometry.x
    triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)
    points = scale * points + shift
    points = points[:, :2]
    basix_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    return dolfinx.mesh.create_mesh(msh.comm, triangles, points, basix_element)


def rescale_mesh_in_place(
    msh: mesh.Mesh,
    scale: float = 1,
    shift: np.ndarray[tuple[int], np.dtype[float]] = np.array([0, 0, 0]),
) -> mesh.Mesh:
    r"""Rescales and shifts the mesh in place.

    Args:
        msh: Mesh
        scale: constant scalign factor
        shift: (3,) shape array containing the shift.

    For example consider the unit-square $[0,1]^2$.
    This is remapped to $\text{scale}*[0, 1]^2 + \text{shift} = [\text{shift}[0], \text{scale}+\text{shift}[1]]^2$.
    This is used to map between the Unit cell $Y=[0,1]^d$ and $Y_\varepsilon$ in the HMM.
    """
    shift = np.asarray(shift)
    assert len(shift.shape) == 1, "Only constant shift allowed, please supply shift with shape (3,)"
    assert shift.shape[0] == 3, "Need 3D shift, i.e. shift.shape=(3,)"
    msh.geometry.x[:] *= scale
    msh.geometry.x[:] += shift
    return msh


def create_periodic_boundary_conditions(
    function_space: fem.FunctionSpace,
    bcs: list[fem.DirichletBC] | None = None,
) -> MultiPointConstraint:
    """Creates periodic boundary condition on the unit square or unit cube.
    For implementation details see
    [`_create_periodic_boundary_conditions_2d`][hommx.helpers._create_periodic_boundary_conditions_2d]
    and
    [`_create_periodic_boundary_conditions_3d`][hommx.helpers._create_periodic_boundary_conditions_3d]
    """
    msh = function_space.mesh
    if msh.topology.dim == 1:
        raise ValueError("Periodic boundary conditions in 1d not implemented.")
    if msh.topology.dim == 2:
        return _create_periodic_boundary_conditions_2d(msh, function_space, bcs)
    if msh.topology.dim == 3:
        return _create_periodic_boundary_conditions_3d(msh, function_space, bcs)
    raise ValueError(
        f"Unkown topology dimension. {function_space.mesh.topology.dim=} is something unexpected"
    )


def _create_periodic_boundary_conditions_2d(
    msh: mesh.Mesh,
    function_space: fem.FunctionSpace,
    bcs: list[fem.DirichletBC] | None = None,
) -> MultiPointConstraint:
    """Creates periodic boundary condition on the unit square using dolfinx_mpc.

    This is done using create_periodic_constraints.
    We do this by forcing $u(x, y_1) = u(x, y_0)$ and $u(x_1, y) = u(x_0, y)$,
    on the unit square we would have $(x_0, x_1) = (y_0, y_1) = (0, 1)$
    Internally we create slave DoFs at $u(x, y_1)$ and $u(x_1, y)$.
    Some attention need to be paid because the node at $(x_1,y_1)$ is constrained twice.
    We use a workaround adapted from: [dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_periodic_gep.py)

    Args:
        msh: Mesh that the boundary conditions should be applied to
        function_space: Function Space on whicht the bc should be applied
        bcs: Dirichlet Boundary Conditions, here the constraint is ignored
    """
    # meshtags used to locate boundary entitites
    if bcs is None:
        bcs = []
    fdim = msh.topology.dim - 1
    TAG_X = 2
    TAG_Y = 3

    # figure out master and slave coordinates, i.e. bottom, top, left, right of box
    slave_coord_x = np.max(msh.geometry.x, axis=0)[0]
    slave_coord_y = np.max(msh.geometry.x, axis=0)[1]

    master_coord_x = np.min(msh.geometry.x, axis=0)[0]
    master_coord_y = np.min(msh.geometry.x, axis=0)[1]

    # marker functions for slaves and master
    def is_slave_x(x):
        return np.isclose(x[0], slave_coord_x)

    def is_slave_y(x):
        return np.isclose(x[1], slave_coord_y)

    # relations between master and slave, i.e. u(x, 1) = u(x, 0)
    def slave_to_master_map_x(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        idx = is_slave_y(x)
        out_x[0][idx] = np.nan  # don't map double constrained points
        return out_x

    def slave_to_master_map_y(x):
        out_x = x.copy()
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        idx = is_slave_x(x)
        out_x[1][idx] = np.nan  # don't map double constrained points
        return out_x

    # locate boundary nodes and tag them
    meshtags = []
    facets_x = mesh.locate_entities_boundary(msh, fdim, is_slave_x)
    facets_y = mesh.locate_entities_boundary(msh, fdim, is_slave_y)
    meshtags = (
        mesh.meshtags(
            msh,
            fdim,
            facets_x[np.argsort(facets_x)],
            np.full(len(facets_x), TAG_X, dtype=np.int32),
        ),
        mesh.meshtags(
            msh,
            fdim,
            facets_y[np.argsort(facets_y)],
            np.full(len(facets_y), TAG_Y, dtype=np.int32),
        ),
    )

    # create multi-point-constraint
    mpc = MultiPointConstraint(function_space)
    # map boundaries
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[0], TAG_X, slave_to_master_map_x, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[1], TAG_Y, slave_to_master_map_y, bcs
    )

    # map (1, 1) to (0, 0)
    def slave_to_master_map(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        idx = np.logical_and(is_slave_x(x), is_slave_y(x))
        out_x[0][~idx] = np.nan
        out_x[1][~idx] = np.nan
        return out_x

    mpc.create_periodic_constraint_topological(
        function_space, meshtags[1], TAG_Y, slave_to_master_map, bcs
    )
    mpc.finalize()
    return mpc


def _create_periodic_boundary_conditions_3d(
    msh: mesh.Mesh,
    function_space: fem.FunctionSpace,
    bcs: list[fem.DirichletBC] | None = None,
) -> MultiPointConstraint:
    """Creates periodic boundary condition on the unit cube using dolfinx_mpc.

    This is done using create_periodic_constraints.
    We do this by forcing $u(x, y_1, z) = u(x, y_0, z)$ and so on for all directions
    on the unit square we would have $(z_0, z_1) = (x_0, x_1) = (y_0, y_1) = (0, 1)$
    Internally we create slave DoFs at $u(x, y_1, z)$ and $u(x_1, y, z)$ etc.
    Some attention need to be paid because the nodes at $(x_1, y_1, z)$ are constrained twice etc for other edges
    And $(x_1, y_1, z_1)$ is constrained three times.
    We use a workaround adapted from: [dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_periodic_gep.py)

    Args:
        msh: Mesh that the boundary conditions should be applied to
        function_space: Function Space on which the bc should be applied
        bcs: Dirichlet Boundary Conditions, here the constraint is ignored
    """
    # meshtags used to locate boundary entitites
    if bcs is None:
        bcs = []
    fdim = msh.topology.dim - 1
    TAG_X = 2
    TAG_Y = 3
    TAG_Z = 4

    # figure out master and slave coordinates, i.e. bottom, top, left, right of box
    slave_coord_x = np.max(msh.geometry.x, axis=0)[0]
    slave_coord_y = np.max(msh.geometry.x, axis=0)[1]
    slave_coord_z = np.max(msh.geometry.x, axis=0)[2]

    master_coord_x = np.min(msh.geometry.x, axis=0)[0]
    master_coord_y = np.min(msh.geometry.x, axis=0)[1]
    master_coord_z = np.min(msh.geometry.x, axis=0)[2]

    # marker functions for slaves and master
    def is_slave_x(x):
        return np.isclose(x[0], slave_coord_x)

    def is_slave_y(x):
        return np.isclose(x[1], slave_coord_y)

    def is_slave_z(x):
        return np.isclose(x[2], slave_coord_z)

    # locate boundary nodes and tag them
    meshtags = []
    facets_x = mesh.locate_entities_boundary(msh, fdim, is_slave_x)
    facets_y = mesh.locate_entities_boundary(msh, fdim, is_slave_y)
    facets_z = mesh.locate_entities_boundary(msh, fdim, is_slave_z)
    meshtags = (
        mesh.meshtags(
            msh,
            fdim,
            facets_x[np.argsort(facets_x)],
            np.full(len(facets_x), TAG_X, dtype=np.int32),
        ),
        mesh.meshtags(
            msh,
            fdim,
            facets_y[np.argsort(facets_y)],
            np.full(len(facets_y), TAG_Y, dtype=np.int32),
        ),
        mesh.meshtags(
            msh,
            fdim,
            facets_z[np.argsort(facets_z)],
            np.full(len(facets_z), TAG_Z, dtype=np.int32),
        ),
    )

    # relations between master and slave that are constrained once, i.e. u(x, 1, z) = u(x, 0, z)
    def slave_to_master_map_x(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        idx = is_slave_y(x) | is_slave_z(x)
        out_x[0][idx] = np.nan  # don't map double constrained points
        return out_x

    def slave_to_master_map_y(x):
        out_x = x.copy()
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        idx = is_slave_x(x) | is_slave_z(x)
        out_x[1][idx] = np.nan  # don't map double constrained points
        return out_x

    def slave_to_master_map_z(x):
        out_x = x.copy()
        out_x[2] = x[2] - (slave_coord_z - master_coord_z)
        idx = is_slave_x(x) | is_slave_y(x)
        out_x[2][idx] = np.nan  # don't map double constrained points
        return out_x

    # create multi-point-constraint
    mpc = MultiPointConstraint(function_space)
    # map boundaries
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[0], TAG_X, slave_to_master_map_x, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[1], TAG_Y, slave_to_master_map_y, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[2], TAG_Z, slave_to_master_map_z, bcs
    )

    # map (1, 1, z) to (0, 0, z), (x, 1, 1) to (x, 0, 0) etc., i.e. double constrained
    def slave_to_master_map_x_y(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        idx = is_slave_x(x) & is_slave_y(x) & ~is_slave_z(x)
        out_x[0][~idx] = np.nan
        out_x[1][~idx] = np.nan
        return out_x

    def slave_to_master_map_x_z(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        out_x[2] = x[2] - (slave_coord_z - master_coord_z)
        idx = is_slave_x(x) & is_slave_z(x) & ~is_slave_y(x)
        out_x[0][~idx] = np.nan
        out_x[2][~idx] = np.nan
        return out_x

    def slave_to_master_map_y_z(x):
        out_x = x.copy()
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        out_x[2] = x[2] - (slave_coord_z - master_coord_z)
        idx = is_slave_y(x) & is_slave_z(x) & ~is_slave_x(x)
        out_x[1][~idx] = np.nan
        out_x[2][~idx] = np.nan
        return out_x

    # map (1,1,1) to (0,0,0), i.e. three times constrained
    def slave_to_master_map_x_y_z(x):
        out_x = x.copy()
        out_x[0] = x[0] - (slave_coord_x - master_coord_x)
        out_x[1] = x[1] - (slave_coord_y - master_coord_y)
        out_x[2] = x[2] - (slave_coord_z - master_coord_z)
        idx = is_slave_y(x) & is_slave_z(x) & is_slave_x(x)
        out_x[0][~idx] = np.nan
        out_x[1][~idx] = np.nan
        out_x[2][~idx] = np.nan
        return out_x

    mpc.create_periodic_constraint_topological(
        function_space, meshtags[0], TAG_X, slave_to_master_map_x_y, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[0], TAG_X, slave_to_master_map_x_z, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[1], TAG_Y, slave_to_master_map_y_z, bcs
    )
    mpc.create_periodic_constraint_topological(
        function_space, meshtags[1], TAG_Y, slave_to_master_map_x_y_z, bcs
    )
    mpc.finalize()
    return mpc


class PoissonFEM:
    r"""solves the poisson equation"""

    def __init__(
        self,
        msh: mesh.Mesh,
        A: Callable[[ufl.SpatialCoordinate], ufl.Form],
        f: ufl.form,
    ):
        r"""Initializes the solver, meshes and boundary condtions.

        Args:
            msh: The macro mesh, on which we want to solve the oscillatory Poisson equation.
                This mesh can live on MPI.COMM_WORLD and the cell problems are automatically solved
                on each process in parallel.
            A: The coefficient, it should be callable like: `A(x)(y)`,
                where $x$ is a spatial coordinate on the macro mesh (the cell center $c_T$)
                and $y$ is a ufl.SpatialCoordinate on the microscopic mesh,
                that is passed to dolfinx to solve the cell problem.
                A needs to be 1-periodic in y, at least for the theory to work.
            f: The right hand side of the Poisson problem.
            msh_micro (mesh.Mesh): The microscopic mesh, this needs to be the unit-square.
                Further it needs to live on MPI.COMM_SELF, since every process owns a whole copy
                of the microscopic mesh. If any other communicator but MPI.COMM_SELF is used the
                results will most likely be rubish.
            eps: $\varepsilon$, the microscopic scaling. Note that this needs to be small enough,
                so that the cells live entirely within their corresponding element.
                If this is not the case, results may be rubish.
            petsc_options_global_solve (optional): PETSc solver options for the global solver, see
            PETSC documentation.
            petsc_options_cell_problem (optional): PETSc solver options for the global solver, see
            PETSC documentation.
            petsc_options_prefix (optional): options prefix used for PETSc options. Defaults to "hommx_PoissonHMM".
        """
        self._msh = msh
        self._comm = msh.comm
        self._coeff = A
        self._f = f
        self._V = fem.functionspace(self._msh, ("Lagrange", 1))  # Macroscopic function space
        self._v_test = ufl.TestFunction(self._V)
        self._u_trial = ufl.TrialFunction(self._V)
        self._x = ufl.SpatialCoordinate(self._msh)
        lhs = ufl.inner(A(self._x) * ufl.grad(self._u_trial), ufl.grad(self._v_test)) * ufl.dx
        rhs = ufl.inner(f(self._x), self._v_test) * ufl.dx
        left = np.min(msh.geometry.x[:, 0])
        right = np.max(msh.geometry.x[:, 0])
        bottom = np.min(msh.geometry.x[:, 1])
        top = np.max(msh.geometry.x[:, 1])
        facets = mesh.locate_entities_boundary(
            msh,
            dim=(msh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], left)
            | np.isclose(x[0], right)
            | np.isclose(x[1], bottom)
            | np.isclose(x[1], top),
        )
        dofs = fem.locate_dofs_topological(
            self._V, entity_dim=(msh.topology.dim - 1), entities=facets
        )
        bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=self._V)
        self._bcs = [bc]
        self._lp = LinearProblem(lhs, rhs, self._bcs)

    def solve(self) -> fem.Function:
        """Assemble the LHS, RHS and solve the problem

        This method assembles the HMM stiffness matrix, so depending on the problem it might
        run for some time.
        """
        self._u = self._lp.solve()
        return self._u

    def plot_solution(self, u: fem.Function | None = None):
        """Simple plot of the solution using pyvista.

        Solve needs to be run before calling this.
        On parallel methods each process only plots the local part.

        """
        if u is None:
            u = self._u
        cells, types, x = plot.vtk_mesh(self._V)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u.x.array
        grid.set_active_scalars("u")
        plotter = pv.Plotter(notebook=True)
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)

        plotter.show()


def plot_fem_function(V: fem.FunctionSpace, u: fem.Function):
    cells, types, x = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array
    grid.set_active_scalars("u")
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)

    plotter.show()
