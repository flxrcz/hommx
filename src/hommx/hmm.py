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
from petsc4py import PETSc
from tqdm import tqdm

import hommx.cell_problem as cell_problem
import hommx.helpers as helpers

REFERENCE_EVALUATION_POINT = np.array([[1 / 3, 1 / 3]])
NUM_BASIS_FUNCTIONS = 3  # number of basis functions on one element


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
    a_H(v_H, w_H) = \sum_{T\in \mathcal T_H} \frac{|T|}{|Y_\varepsilon(c_T)|} \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla R_{T, h}(v_h)\cdot \nabla R_{T, h}(w_h) dx,
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
        self._cell_mesh = helpers.rescale_mesh(msh_micro)  # create a copy of the mesh, just in case
        self._V_macro = fem.functionspace(self._msh, ("Lagrange", 1))  # Macroscopic function space
        self._v_test = ufl.TestFunction(self._V_macro)
        self._x = ufl.SpatialCoordinate(self._msh)
        L = ufl.inner(f(self._x), self._v_test) * ufl.dx
        self._L = fem.form(L)
        self._b = create_vector(self._L)
        self._u = fem.Function(self._V_macro)
        self._x = la.create_petsc_vector_wrap(self._u.x)

        # Dirichlet BC
        left = np.min(self._msh.geometry.x[:, 0])
        right = np.max(self._msh.geometry.x[:, 0])
        bottom = np.min(self._msh.geometry.x[:, 1])
        top = np.max(self._msh.geometry.x[:, 1])
        facets = mesh.locate_entities_boundary(
            self._msh,
            dim=(self._msh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], left)
            | np.isclose(x[0], right)
            | np.isclose(x[1], bottom)
            | np.isclose(x[1], top),
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

        self._setup_cell_problems()
        num_local_cells = self._V_macro.mesh.topology.index_map(2).size_local

        # cell problem loop
        for local_cell_index in tqdm(range(num_local_cells)):
            local_dofs = self._V_macro.dofmap.cell_dofs(local_cell_index)
            global_dofs = self._V_macro.dofmap.index_map.local_to_global(local_dofs).astype(
                PETSc.IntType
            )
            # local assembly for one cell
            S_loc = self._compute_local_stiffness(local_cell_index)
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

    def _setup_cell_problems(self) -> None:
        """Set up various structures used throughout all cell problems.

        Since all ufl forms are precompiled the mathematical description of the cell problem
        and stiffness assembly are in this function.
        If you want to adapt the HMM to your problem, you can simply overwrite this method and
        change the ufl forms.

        Notes:
            This method sets up objects that are used through the cell problems:
            - function spaces and functions on the cell mesh
            - periodic boundary conditions
            - ufl forms containg the cell problem and local stiffness assembly
                - corresponding constants
        """
        # micro function space and periodic boundary conditions
        self._V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1))
        self._mpc = helpers.create_periodic_boundary_conditions(self._V_micro, self._bcs)
        self._points_micro = self._V_micro.tabulate_dof_coordinates()
        self._v_tilde = ufl.TrialFunction(self._V_micro)
        self._z = ufl.TestFunction(self._V_micro)
        self._y = ufl.SpatialCoordinate(self._cell_mesh)
        self._grad_v_micro = fem.Constant(self._cell_mesh, np.zeros((2,)))
        # wrap x in A(x, y) in a Constant to avoid recompilation
        self._x_macro = fem.Constant(
            self._cell_mesh, np.zeros((self._cell_mesh.geometry.x.shape[1],))
        )
        self._A_micro = self._coeff(self._x_macro, self._y)
        # precompile cell problem LHS and RHS
        self._a_micro_compiled = fem.form(
            ufl.inner(self._A_micro * ufl.grad(self._v_tilde), ufl.grad(self._z)) * ufl.dx
        )
        self._L_micro_compiled = fem.form(
            -ufl.inner(self._A_micro * self._grad_v_micro, ufl.grad(self._z)) * ufl.dx
        )
        # setup of macro functions once for all cell problems
        self._v_macro = fem.Function(self._V_macro)
        self._grad_v_macro = fem.Expression(ufl.grad(self._v_macro), REFERENCE_EVALUATION_POINT)
        # setup of placeholder functions for the micro functions
        self._v_micros = [fem.Function(self._V_micro) for _ in range(NUM_BASIS_FUNCTIONS)]
        # placeholders for the correctors
        self._correctors = [fem.Function(self._V_micro) for _ in range(NUM_BASIS_FUNCTIONS)]
        self._local_stiffness_forms = [
            [
                fem.form(
                    ufl.inner(
                        self._A_micro
                        * (ufl.grad(self._v_micros[i]) + ufl.grad(self._correctors[i])),
                        ufl.grad(self._v_micros[j]) + ufl.grad(self._correctors[j]),
                    )
                    * ufl.dx
                )
                for i in range(NUM_BASIS_FUNCTIONS)
            ]
            for j in range(NUM_BASIS_FUNCTIONS)
        ]

    def _compute_local_stiffness(
        self, cell_index: int
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """Computes the local stiffness matrix on one element by solving the cell problem for all
        basis functions on that cell.
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
        # update the x value in the precompiled forms containing A
        self._x_macro.value = c_t

        for i, local_dof in enumerate(local_dofs):
            self._v_macro.x.array[:] = 0.0
            self._v_macro.x.array[local_dof] = 1.0
            self._interpolate_macro_to_micro(
                self._v_macro, self._grad_v_macro, cell_index, self._v_micros[i]
            )
            self._calculate_corrector(
                cell_index,
                self._correctors[i],
            )

        # local stiffness matrix
        S_loc = np.zeros((NUM_BASIS_FUNCTIONS, NUM_BASIS_FUNCTIONS))
        for i in range(S_loc.shape[0]):
            for j in range(S_loc.shape[1]):
                S_loc[i, j] = fem.assemble_scalar(self._local_stiffness_forms[i][j])

        # scale contribution
        cell_area = _triangle_area(points)
        Y_area = 1
        return S_loc * cell_area / Y_area

    def _interpolate_macro_to_micro(
        self,
        v_macro: fem.Function,
        grad_v_macro_expr: fem.Expression,
        cell_index: int,
        v_micro: fem.Function = None,
    ) -> fem.Function:
        """Interpolates a function from the macro mesh onto the micro mesh.

        Since we know that the micro domain is contained inside one macro cell,
        we can avoid dolfinx interpolate_nonmatching and instead rely on evaluating directly.

        Notes:
            This function has the side aeffect of updating the constant self._grad_v_micro
            that is used in the precompiled form for the cell problem.

        Args:
            v_macro: Macro function
            grad_v_macro_expr: Expression for the gradient of the macroscopic function
            cell_index: cell index on which the cell problem is solved
            v_micro (optional): Micro function, if none is provided one is created

        """
        if v_micro is None:
            v_micro = fem.Function(self._V_micro)
        # v_micro.interpolate_nonmatching(v_macro, interpolation_cells, interpolation_data) # DOES NOT WORK IN PARALLEL
        cells = np.full(self._points_micro.shape[0], cell_index, dtype=np.int32)
        v_micro.x.array[:] = v_macro.eval(self._points_micro, cells=cells).flatten()

        # update gradient constant in compiled form
        self._grad_v_micro.value = grad_v_macro_expr.eval(self._msh, [cell_index]).flatten()
        return v_micro

    def _calculate_corrector(
        self,
        cell_index: int,
        corrector: fem.Function = None,
    ):
        """Calculates the corrector by solving the cell problem.

        Notes:
            The micro function for which the corrector should be calculated does not show up,
            since the precompiled form uses a contant that is updated in _interpolate_macro_to_micro.

        Args:
            cell_index: the process-local index of the cell
                on which the cell problem should be solved
            corrector (optional): function on the micro mesh that stores the corrector,
                if none is provided one is created
        """
        if corrector is None:
            corrector = fem.Function(self._V_micro)

        problem = cell_problem.PeriodicLinearProblem(
            self._a_micro_compiled,
            self._L_micro_compiled,
            self._mpc,
            petsc_options=self._petsc_options_cell_problem,
        )
        v_tilde_sol = problem.solve()
        if problem._solver.getConvergedReason() < 0:
            self._logger.error(
                f"Something went wrong in the cell problem solving for cell {cell_index}. PETSc solver failed with reason {problem._solver.getConvergedReason()}"
            )
        corrector.x.array[:] = v_tilde_sol.x.array
        return corrector

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

    def plot_solution(self, u: fem.Function | None = None):
        """Simple plot of the solution using pyvista.

        Solve needs to be run before calling this.
        On parallel methods each process only plots the local part.

        """
        if u is None:
            u = self._u
        cells, types, x = plot.vtk_mesh(self._V_macro)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u.x.array
        grid.set_active_scalars("u")
        plotter = pv.Plotter(notebook=True)
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)

        plotter.show()


class PoissonSemiHMM(PoissonHMM):
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
    a_H(v_H, w_H) = \sum_{T\in \mathcal T_H} \frac{|T|}{|Y_\varepsilon(c_T)|} \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla R_{T, h}(v_h)\cdot \nabla R_{T, h}(w_h) dx,
    $$

    where $\nabla R_{T, h} = v_H|_{Y_\varepsilon(c_T)} + D\theta(x)\tilde{v_h}, \tilde{v_h}\in V_{h, \#}(Y_\varepsilon(c_T))$ is the reconstruction operator,
    where $\tilde{v_h}$ is the solution to

    $$
    \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) D\theta(x)\nabla\tilde{v_h} \cdot D\theta(x)\nabla z_h dx = - \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) \nabla v_H \cdot D\theta(x)\nabla z_h dx
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
        A: Callable[[fem.Constant, ufl.SpatialCoordinate], ufl.Form],
        f: ufl.form,
        msh_micro: mesh.Mesh,
        eps: float,
        Dtheta: Callable[[fem.Constant], ufl.Form],
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_PoissonHMM",
    ):
        super().__init__(
            msh,
            A,
            f,
            msh_micro,
            eps,
            petsc_options_global_solve,
            petsc_options_cell_problem,
            petsc_options_prefix,
        )
        self._Dtheta = Dtheta

    def _setup_cell_problems(self) -> None:
        """Set up various structures used throughout all cell problems.

        Since all ufl forms are precompiled the mathematical description of the cell problem
        and stiffness assembly are in this function.
        If you want to adapt the HMM to your problem, you can simply overwrite this method and
        change the ufl forms.

        Notes:
            This method sets up objects that are used through the cell problems:
            - function spaces and functions on the cell mesh
            - periodic boundary conditions
            - ufl forms containg the cell problem and local stiffness assembly
                - corresponding constants
        """
        # micro function space and periodic boundary conditions
        self._V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1))
        self._mpc = helpers.create_periodic_boundary_conditions(self._V_micro, self._bcs)
        self._points_micro = self._V_micro.tabulate_dof_coordinates()
        self._v_tilde = ufl.TrialFunction(self._V_micro)
        self._z = ufl.TestFunction(self._V_micro)
        self._y = ufl.SpatialCoordinate(self._cell_mesh)
        self._grad_v_micro = fem.Constant(self._cell_mesh, np.zeros((2,)))
        # wrap x in A(x, y) in a Constant to avoid recompilation
        self._x_macro = fem.Constant(
            self._cell_mesh, np.zeros((self._cell_mesh.geometry.x.shape[1],))
        )
        self._A_micro = self._coeff(self._x_macro, self._y)
        self._Dthetax = self._Dtheta(self._x_macro)
        # precompile cell problem LHS and RHS
        self._a_micro_compiled = fem.form(
            ufl.inner(
                self._A_micro * self._Dthetax * ufl.grad(self._v_tilde),
                self._Dthetax * ufl.grad(self._z),
            )
            * ufl.dx
        )
        self._L_micro_compiled = fem.form(
            -ufl.inner(self._A_micro * self._grad_v_micro, self._Dthetax * ufl.grad(self._z))
            * ufl.dx
        )
        # setup of macro functions once for all cell problems
        self._v_macro = fem.Function(self._V_macro)
        self._grad_v_macro = fem.Expression(ufl.grad(self._v_macro), REFERENCE_EVALUATION_POINT)
        # setup of placeholder functions for the micro functions
        self._v_micros = [fem.Function(self._V_micro) for _ in range(NUM_BASIS_FUNCTIONS)]
        # placeholders for the correctors
        self._correctors = [fem.Function(self._V_micro) for _ in range(NUM_BASIS_FUNCTIONS)]
        self._local_stiffness_forms = [
            [
                fem.form(
                    ufl.inner(
                        self._A_micro
                        * (
                            ufl.grad(self._v_micros[i])
                            + self._Dthetax * ufl.grad(self._correctors[i])
                        ),
                        (
                            ufl.grad(self._v_micros[j])
                            + self._Dthetax * ufl.grad(self._correctors[j])
                        ),
                    )
                    * ufl.dx
                )
                for i in range(NUM_BASIS_FUNCTIONS)
            ]
            for j in range(NUM_BASIS_FUNCTIONS)
        ]


def _triangle_area(points):
    return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
