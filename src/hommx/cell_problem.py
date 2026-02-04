"""Helper problems that are used to solve the cell problem in the HMM.

Classes:
    PeriodicLinearProblem: linear problem class that automatically
        adds periodic boundary conditions on boxes.
"""

import dolfinx_mpc
import numpy as np
from dolfinx import fem, mesh
from dolfinx_mpc.assemble_matrix import assemble_matrix
from dolfinx_mpc.assemble_vector import apply_lifting, assemble_vector
from petsc4py import PETSc


def create_periodic_boundary_conditions(
    function_space: fem.FunctionSpace,
    bcs: list[fem.DirichletBC] | None = None,
) -> dolfinx_mpc.MultiPointConstraint:
    """Creates periodic boundary condition on the unit square or unit cube.
    For implementation details see
    [`_create_periodic_boundary_conditions_2d`][hommx.cell_problem._create_periodic_boundary_conditions_2d]
    and
    [`_create_periodic_boundary_conditions_3d`][hommx.cell_problem._create_periodic_boundary_conditions_3d]
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
) -> dolfinx_mpc.MultiPointConstraint:
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
    mpc = dolfinx_mpc.MultiPointConstraint(function_space)
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
) -> dolfinx_mpc.MultiPointConstraint:
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
    mpc = dolfinx_mpc.MultiPointConstraint(function_space)
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


class PeriodicLinearProblem(dolfinx_mpc.LinearProblem):
    """
    Class for solving a linear variational problem with periodic boundary conditions
    with multi point constraints of the form
    $a(u, v) = L(v)$ for all v using PETSc as a linear algebra backend.

    The solution is only unique up to a constant.
    This is handled by telling the PETSc KSP solver about the nullspace.

    Args:
        a: A bilinear UFL form, the left hand side of the variational problem.
        L: A linear UFL form, the right hand side of the variational problem.
        mpc: The multi point constraint.
        bcs: A list of Dirichlet boundary conditions.
        u: The solution function. It will be created if not provided. The function has
            to be based on the functionspace in the mpc, i.e.

            .. highlight:: python
            .. code-block:: python

                u = dolfinx.fem.Function(mpc.function_space)
        petsc_options: Parameters that is passed to the linear algebra backend PETSc.  #type: ignore
            For available choices for the 'petsc_options' kwarg, see the PETSc-documentation
            https://www.mcs.anl.gov/petsc/documentation/index.html.
        form_compiler_options: Parameters used in FFCx compilation of this form. Run `ffcx --help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by DOLFINx.
        jit_options: Parameters used in CFFI JIT compilation of C code generated by FFCx.
            See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py#L22-L37
            for all available parameters. Takes priority over all other parameter values.
    Examples:
        Example usage:

        .. highlight:: python
        .. code-block:: python

           problem = LinearProblem(
               a, L, mpc, [bc0, bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
           )

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "_A")
        comm = self._a.mesh.comm
        nullspace_vector = PETSc.Vec().create(comm)
        nullspace_vector.setSizes(self.A.getSize()[0])
        nullspace_vector.setUp()
        nullspace_vector.set(1.0)
        nullspace_vector.setValues(self._mpc.slaves, np.zeros_like(self._mpc.slaves))
        nullspace_vector.assemble()
        nullspace = PETSc.NullSpace().create(vectors=(nullspace_vector,), comm=comm)
        # self._A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self._A.setNullSpace(nullspace)
        self._A.setNearNullSpace(nullspace)
        self._solver.setOperators(self._A)
        nullspace.remove(self._b)
        self._nullspace = nullspace

    def solve(self) -> fem.Function:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix(self._a, self._mpc, bcs=self.bcs, A=self._A)
        self._A.assemble()
        assert self._A.assembled

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._L, self._mpc, b=self._b)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], [self.bcs], self._mpc)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        fem.petsc.set_bc(self._b, self.bcs)

        self._nullspace.remove(self._b)
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()
        self._mpc.backsubstitution(self.u)

        return self.u
