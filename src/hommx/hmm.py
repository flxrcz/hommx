"""Implementation of the Heterogenous-Multi-Scale-Method using dolfinx.

Classes:
    PoissonHMM: Solver for the HMM on the Poisson Equation.
"""

import logging
from collections.abc import Callable

import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, la, mesh, plot
from dolfinx.fem.assemble import _assemble_vector_array
from dolfinx.fem.petsc import create_vector
from dolfinx.geometry import PointOwnershipData
from petsc4py import PETSc
from tqdm import tqdm

import hommx.cell_problem as cell_problem
import hommx.helpers as helpers

REFERENCE_EVALUATION_POINT = np.array([[1 / 3, 1 / 3]])


class PoissonHMM:
    r"""Solver for the Multi-Scale Poisson problem using the HMM.

    This class implements the Heterogenous-Multi-Scale Method for a Poisson problem.
    We want to solve the weak formulation of the Poisson problem:

    $$
        \int_\Omega (A\nabla u) \cdot \nabla v dx = \int_\Omega fv dx
    $$

    With Dirichlet-Boundary Conditions $u=0$ on $\partial\Omega$.

    We do this by approximating the homogenized coefficient on every cell and
    using the adapted bilinear form

    $$
    a_H(v_H, w_H) = \sum_{T\in \mathcal T_H} \frac{|T|}{|Y_\varepsilon(c_T)|} \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla R_{T, h}(v_h)\cdot \nabla R_{T, h} dx,
    $$

    where $R_{T, h} = v_H|_{Y_\varepsilon(c_T)} + \tilde{v_h}, \tilde{v_h}\in V_{h, \#}(Y_\varepsilon(c_T))$ is the reconstruction operator,
    where $\tilde{v_h}$ is the solution to

    $$
    \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla\tilde{v_h} \cdot \nabla z_h dx = - \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla v_H \cdot \nabla z_h dx
    $$

    note that the gradient of the macro-scale function $v_H$ appears on the RHS.

    $Y_\varepsilon(c_T) = c_T + [-\varepsilon/2, \varepsilon]^d$ is the micro mesh cell.

    Parallelization:
    The code can in theory run in parallel.
    If you run the code in serial, the setup of the meshes can be arbitrary.
    If you want to run the code in parallel, it is only supported for now that the micro mesh lives
    on MPI.COMM_SELF. Parallelization is done by each process solving the cell problems for it's local
    part.
    Passing a mesh that lives on anything but MPI.COMM_SELF to the msh_micro parameter can lead to
    unexpected behaviour.


    Notes:
        - For now only zero-Dirichlet Boundary Conditions are implemented.
        - It is the users responsibility to ensure that the micro meshes fit into the macro mesh cells.
        I.e. the shifted and scaled versions of $Y$ $Y_\varepsilon(c_T)$ need to fit within the element $T$.
        Otherwise the interpolation of the macro scale basis functions to the micro scale may lead to
        unexpected behaviour.
    """

    def __init__(
        self,
        msh: mesh.Mesh,
        A: Callable[
            [np.ndarray[tuple[int], np.dtype[float]]], Callable[[ufl.SpatialCoordinate], ufl.Form]
        ],
        f: ufl.form,
        msh_micro: mesh.Mesh,
        eps: float,
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_PoissonHMM",
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
        self._eps = eps
        self._cell_mesh = helpers.rescale_mesh(
            msh_micro
        )  # create a copy of the mesh, since we modify it
        self._V_macro = fem.functionspace(self._msh, ("Lagrange", 1))  # Macroscopic function space
        self._u = ufl.TrialFunction(self._V_macro)
        self._v = ufl.TestFunction(self._V_macro)
        self._x = ufl.SpatialCoordinate(self._msh)
        L = ufl.inner(f(self._x), self._v) * ufl.dx
        self._L = fem.form(L)
        self._b = create_vector(self._L)
        self._u = fem.Function(self._V_macro)
        self._x = la.create_petsc_vector_wrap(self._u.x)

        # Dirichlet BC
        facets = mesh.locate_entities_boundary(
            self._msh,
            dim=(self._msh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0),
        )
        dofs = fem.locate_dofs_topological(self._V_macro, entity_dim=1, entities=facets)
        bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=self._V_macro)
        self._bcs = [bc]
        self._num_dofs = self._V_macro.dofmap.index_map.size_global
        self._num_local_dofs = self._V_macro.dofmap.index_map.size_local
        self._A = PETSc.Mat().createAIJ(
            ((self._num_local_dofs, self._num_dofs), (self._num_local_dofs, self._num_dofs))
        )
        self._logger = logging.getLogger(__name__)

        if petsc_options_cell_problem is None:
            petsc_options_cell_problem = {
                "ksp_atol": 1e-12,  # if this is too low than the solver will get stuck on non-periodic problems
            }
        self._petsc_options_cell_problem = petsc_options_cell_problem
        self._solver = PETSc.KSP().create(self._comm)
        if petsc_options_global_solve is not None:
            opts = PETSc.Options()

            for k, v in petsc_options_global_solve.items():
                opts[k] = v

            self._solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options_global_solve.keys():
                del opts[k]

    def _assemble_stiffness(self):
        """Assembly of the stiffness matrix in parallel."""
        if self._A.assembled:
            return
        # set up variables that can be reused across all cell solves
        self._old_c_t = None
        # We set up the periodic boundary conditions once.
        # In theory and practice this should be fine, since the meshes are just scaled.
        # In practice this may end up failing once, because dolfinx_mpc only guarantees
        # that the boundary conditions work well on the function space they were created on.
        V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1))
        self._mpc = helpers.create_periodic_boundary_conditions(self._cell_mesh, V_micro, self._bcs)

        num_local_cells = self._V_macro.mesh.topology.index_map(2).size_local
        # setup of macro functions once on each process
        self._v_macro = fem.Function(self._V_macro)
        self._grad_v_macro = fem.Expression(ufl.grad(self._v_macro), REFERENCE_EVALUATION_POINT)
        # cell problem loop
        for local_cell_index in tqdm(range(num_local_cells)):
            local_dofs = self._V_macro.dofmap.cell_dofs(local_cell_index)
            global_dofs = self._V_macro.dofmap.index_map.local_to_global(local_dofs).astype(
                PETSc.IntType
            )
            # local assembly for one cell
            S_loc = self._solve_cell_problem(local_cell_index)
            if np.any(np.isnan(S_loc)):
                self._logger.error(
                    f"Something went wrong when calculating local matrix on cell {local_cell_index}"
                )
            # global assembly
            self._A.setValues(
                global_dofs, global_dofs, S_loc.flatten(), addv=PETSc.InsertMode.ADD_VALUES
            )

        # enforce Dirichlet BC
        bc_dofs, ghost_index = self._bcs[0].dof_indices()
        bc_global_dofs = self._V_macro.dofmap.index_map.local_to_global(
            bc_dofs[:ghost_index]
        ).astype(PETSc.IntType)
        self._A.assemble()
        self._A.zeroRowsColumns(bc_global_dofs, diag=1.0)

    def _solve_cell_problem(self, cell_index: int) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """Solves the cell problem on one cell.
        All computation is done on one process and no communication takes place between the processes.

        Args:
            cell_index: process-local index of the cell for which the homogenized coefficient
            is to be approximated

        Returns:
            np.ndarray: local stiffness matrix
        """
        local_dofs = self._V_macro.dofmap.cell_dofs(cell_index)
        points = self._V_macro.tabulate_dof_coordinates()[local_dofs]
        c_t = np.mean(points, axis=0)

        # set up micro function space
        V_micro = self._setup_micro_mesh(c_t)
        v_micro_list = []
        grad_v_micro_list = []

        coords_micro = V_micro.tabulate_dof_coordinates()
        _points = coords_micro

        global_dofs = self._V_macro.dofmap.index_map.local_to_global(local_dofs)
        for local_dof, global_dof in zip(local_dofs, global_dofs):
            # set up macro basis function
            # this only changes the local copy including potentially ghost nodes, so no communication is needed
            self._v_macro.x.array[:] = 0.0
            self._v_macro.x.array[local_dof] = 1.0

            # interpolate basis function on micro space
            v_micro = fem.Function(V_micro)
            # v_micro.interpolate_nonmatching(v_macro, interpolation_cells, interpolation_data) # DOES NOT WORK IN PARALLEL
            cells = np.full(_points.shape[0], cell_index, dtype=np.int32)
            vals = self._v_macro.eval(_points, cells=cells)
            vals = np.asarray(vals).flatten()

            v_micro = fem.Function(V_micro)
            v_micro.x.array[:] = vals
            v_micro_list.append(v_micro)

            # evaluate gradient
            grad_v_micro_list.append(self._grad_v_macro.eval(self._msh, [cell_index]).flatten())

        # solve cell_problems
        v_tilde_list = []  # correctors
        v_tilde = ufl.TrialFunction(V_micro)
        z = ufl.TestFunction(V_micro)
        y = ufl.SpatialCoordinate(self._cell_mesh)
        A_micro = self._coeff(c_t)(y / self._eps)
        for i in range(3):
            grad_v_micro = fem.Constant(self._cell_mesh, grad_v_micro_list[i])
            a = ufl.inner(A_micro * ufl.grad(v_tilde), ufl.grad(z)) * ufl.dx
            L = -ufl.inner(A_micro * grad_v_micro, ufl.grad(z)) * ufl.dx
            problem = cell_problem.PeriodicLinearProblem(
                a, L, self._mpc, petsc_options=self._petsc_options_cell_problem
            )
            v_tilde_sol = problem.solve()
            if problem._solver.getConvergedReason() < 0:
                self._logger.error(
                    f"Something went wrong in the cell problem solving for cell {cell_index}. PETSc solver failed with reason {problem._solver.getConvergedReason()}"
                )
            v_tilde_list.append(v_tilde_sol)

        # build reconstruction operator R_T
        R_list = []
        for i in range(3):
            R_i = fem.Function(V_micro)
            R_i.x.array[:] = v_micro_list[i].x.array + v_tilde_list[i].x.array
            R_list.append(R_i)
        # local stiffness matrix
        S_loc = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                S_loc[i, j] = fem.assemble_scalar(
                    fem.form(ufl.inner(A_micro * ufl.grad(R_list[i]), ufl.grad(R_list[j])) * ufl.dx)
                )

        # scale contribution
        cell_area = _triangle_area(points)
        Y_eps_area = self._eps**2
        return S_loc * cell_area / Y_eps_area

    def _setup_micro_mesh(
        self, c_t: np.ndarray[tuple[int], np.dtype[float]]
    ) -> tuple[fem.FunctionSpace, np.ndarray[tuple[int], np.dtype[int]], PointOwnershipData]:
        if self._old_c_t is None:
            self._cell_mesh = helpers.rescale_mesh(
                self._cell_mesh, self._eps, c_t - self._eps * np.array([1 / 2, 1 / 2, 0])
            )
        else:
            self._cell_mesh = helpers.rescale_mesh_in_place(
                self._cell_mesh, scale=1, shift=-self._old_c_t + c_t
            )
        self._old_c_t = c_t
        V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1))
        return V_micro

    def solve(self) -> fem.Function:
        """Assemble the LHS, RHS and solve the problem

        This method assembles the HMM stiffness matrix, so depending on the problem it might
        run for some time.
        """
        # assemble LHS matrix
        self._assemble_stiffness()

        # assemble rhs
        with self._b.localForm() as b_local:
            b_local.set(0)
        with self._b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, self._L, None, None)
        self._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )  # accumulate ghost values on owning process

        # enforce dirichlet boundary 0
        local_bc_dofs = self._bcs[0].dof_indices()[0]
        global_bc_dofs = self._V_macro.dofmap.index_map.local_to_global(local_bc_dofs).astype(
            PETSc.IntType
        )

        self._b.setValues(global_bc_dofs, np.zeros_like(global_bc_dofs))
        self._b.assemble()

        # we don't do lifting for now and instead rely on 0 dirchilet BC
        # with self._b.localForm() as b_local:
        #     apply_lifting(b_local.array_w, self._a, [self._bcs], [], 1, None, None)

        self._solver.setOperators(self._A)

        self._solver.solve(self._b, self._x)
        if self._solver.getConvergedReason() < 0:
            self._logger.error(
                f"Something went wrong in the global problem solve. PETSc solver failed with reason {self._solver.getConvergedReason()}"
            )
        self._u.x.scatter_forward()  # make sure ghosts are updated so plotting works correctly
        return self._u

    def plot_solution(self):
        """Simple plot of the solution using pyvista.

        Solve needs to be run before calling this.
        On parallel methods each process only plots the local part.

        """
        cells, types, x = plot.vtk_mesh(self._V_macro)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = self._u.x.array
        grid.set_active_scalars("u")
        plotter = pv.Plotter(notebook=True)
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        plotter.show()


def _triangle_area(points):
    return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
