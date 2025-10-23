"""Small helper functions used throughout the library

functions:
    mesh_from_file: Reads a mesh from a file to be used in dolfinx.
    rescale_mesh: copies and rescales/shifts a mesh
    rescale_mesh_in_place: rescales/shifts a mesh in place.
    create_periodic_boundary_conditions: creates periodic boundary conditions on a 2D box mesh.
"""

import basix.ufl
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx_mpc import MultiPointConstraint
from mpi4py import MPI
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
    This is remapped to $scale*[0, 1]^2 + shift = [shift[0], scale+shift[1]]^2$.
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
    This is remapped to $scale*[0, 1]^2 + shift = [shift[0], scale+shift[1]]^2$.
    This is used to map between the Unit cell $Y=[0,1]^d$ and $Y_\varepsilon$ in the HMM.
    """
    shift = np.asarray(shift)
    assert len(shift.shape) == 1, "Only constant shift allowed, please supply shift with shape (3,)"
    assert shift.shape[0] == 3, "Need 3D shift, i.e. shift.shape=(3,)"
    msh.geometry.x[:] *= scale
    msh.geometry.x[:] += shift
    return msh


def create_periodic_boundary_conditions(
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
    We use a workaround adapted from: https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_periodic_gep.py
    """
    # meshtags used to locate boundary entitites
    if bcs is None:
        bcs = []
    fdim = msh.topology.dim - 1
    TAG_X = 2
    TAG_Y = 3
    slave_coord_x = 1  # the x coordinate at the bottom of the rectangular mesh
    master_coord_x = 0  # the x coordinate at the top of the rectangular mesh

    slave_coord_y = 1  # the y coordinate at the bottom of the rectangular mesh
    master_coord_y = 0  # the y coordinate at the top of the rectangular mesh

    # figure out
    slave_coord_x = np.min(msh.geometry.x, axis=0)[0]
    slave_coord_y = np.min(msh.geometry.x, axis=0)[1]

    master_coord_x = np.max(msh.geometry.x, axis=0)[0]
    master_coord_y = np.max(msh.geometry.x, axis=0)[1]

    # marker functions for slaves and master
    def is_slave_x(x):
        return np.isclose(x[0], slave_coord_x)

    def is_slave_y(x):
        return np.isclose(x[1], slave_coord_y)

    def is_master_x(x):
        return np.isclose(x[0], master_coord_x)

    def is_master_y(x):
        return np.isclose(x[1], master_coord_y)

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
