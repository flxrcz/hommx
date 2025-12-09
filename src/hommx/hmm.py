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

REFERENCE_EVALUATION_POINT_3D = np.array([[1 / 3, 1 / 3, 1 / 3]])
REFERENCE_EVALUATION_POINT_2D = np.array([[1 / 3, 1 / 3]])


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
        self._cell_mesh = msh_micro  # create a copy of the mesh, just in case
        self._V_macro = fem.functionspace(self._msh, ("Lagrange", 1))  # Macroscopic function space
        self._v_test = ufl.TestFunction(self._V_macro)
        self._x = ufl.SpatialCoordinate(self._msh)
        L = ufl.inner(f(self._x), self._v_test) * ufl.dx
        self._L = fem.form(L)
        self._b = create_vector(self._L)
        self._u = fem.Function(self._V_macro)
        self._x = la.create_petsc_vector_wrap(self._u.x)

        # setup 2D vs 3D differences
        self._tdim = self._msh.topology.dim
        if self._tdim not in (2, 3):
            raise ValueError("Topology should be 3D or 2D")
        if self._tdim != self._msh.geometry.dim:
            raise ValueError(
                "Topological dimension is different from geometrical dimension. Currently surfaces in 3D are not supported."
            )

        self._num_basis_functions = (
            self._tdim + 1
        )  # 3 basis functions for triangles, 4 for tetrahedra
        if self._tdim == 3:
            # Dirichlet BC
            left = np.min(self._msh.geometry.x[:, 0])
            right = np.max(self._msh.geometry.x[:, 0])
            bottom = np.min(self._msh.geometry.x[:, 1])
            top = np.max(self._msh.geometry.x[:, 1])
            back = np.min(self._msh.geometry.x[:, 2])
            front = np.max(self._msh.geometry.x[:, 2])
            facets = mesh.locate_entities_boundary(
                self._msh,
                dim=(self._msh.topology.dim - 1),
                marker=lambda x: np.isclose(x[0], left)
                | np.isclose(x[0], right)
                | np.isclose(x[1], bottom)
                | np.isclose(x[1], top)
                | np.isclose(x[2], back)
                | np.isclose(x[2], front),
            )
            self._reference_evaluation_point = REFERENCE_EVALUATION_POINT_3D
            self._volume_function = _tetrahedron_volume

        if self._tdim == 2:
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
            self._reference_evaluation_point = REFERENCE_EVALUATION_POINT_2D
            self._volume_function = _triangle_area

        dofs = fem.locate_dofs_topological(
            self._V_macro, entity_dim=(self._msh.topology.dim - 1), entities=facets
        )
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

    @property
    def function_space(self) -> fem.FunctionSpace:
        """Function space of the macro mesh, that can be used to set Dirichlet BCs."""
        return self._V_macro

    def set_boundary_condtions(self, bcs: list[fem.DirichletBC] | fem.DirichletBC):
        """Set new boundary conditions. This method needs to be called before solve to accomodate
        the new boundary conditions, by default Dirichlet 0 on the whole boundary is enforced
        for squares or cubes.
        You can extract

        This method needs to be called if the geometry is not a square or cube.
        Args:
            bcs (list[fem.DirichletBC]): list of boundary conditions
        """
        if isinstance(bcs, list):
            self._bcs = bcs
        else:
            self._bcs = [bcs]

    def _assemble_stiffness(self):
        """Assembly of the stiffness matrix in parallel."""
        if self._A.assembled:
            return

        self._setup_cell_problems()
        num_local_cells = self._V_macro.mesh.topology.index_map(self._tdim).size_local

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
            - ufl forms containing the cell problem and local stiffness assembly
                - corresponding constants
        """
        # micro function space and periodic boundary conditions
        self._V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1))
        self._mpc = helpers.create_periodic_boundary_conditions(self._V_micro, self._bcs)
        self._points_micro = self._V_micro.tabulate_dof_coordinates()
        self._v_tilde = ufl.TrialFunction(self._V_micro)
        self._z = ufl.TestFunction(self._V_micro)
        self._y = ufl.SpatialCoordinate(self._cell_mesh)
        self._grad_v_micro = fem.Constant(self._cell_mesh, np.zeros((self._tdim,)))
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
        self._grad_v_macro = fem.Expression(
            ufl.grad(self._v_macro), self._reference_evaluation_point
        )
        # setup of placeholder functions for the micro functions
        self._v_micros = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
        # placeholders for the correctors
        self._correctors = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
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
                for i in range(self._num_basis_functions)
            ]
            for j in range(self._num_basis_functions)
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
        S_loc = np.zeros((self._num_basis_functions, self._num_basis_functions))
        for i in range(S_loc.shape[0]):
            for j in range(S_loc.shape[1]):
                S_loc[i, j] = fem.assemble_scalar(self._local_stiffness_forms[i][j])

        # scale contribution
        cell_area = self._volume_function(points)
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
        self._A.assemble()

        # assemble rhs
        with self._b.localForm() as b_local:
            b_local.set(0)
        with self._b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, self._L, None, None)
        self._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )  # accumulate ghost values on owning process
        self._b.assemble()

        # enforce Dirichlet BC by lifting
        for bc in self._bcs:
            local_bc_dofs, ghost_index = bc.dof_indices()  # process local
            global_bc_dofs = self._V_macro.dofmap.index_map.local_to_global(
                local_bc_dofs[:ghost_index]
            ).astype(PETSc.IntType)
            # create vector for lifting
            u_bc = self._u.copy()
            u_bc.x.array[:] = 0.0
            u_bc.x.array[local_bc_dofs] = bc.g.value
            u_bc.x.scatter_forward()
            b_lift = self._A.createVecLeft()
            self._A.mult(u_bc.x.petsc_vec, b_lift)

            bc.set(self._b)
            self._b.axpy(-1, b_lift)
            self._A.zeroRowsColumns(global_bc_dofs, diag=1.0)
            self._b.setValues(global_bc_dofs, np.full(global_bc_dofs.shape, bc.g.value))
            self._b.assemble()

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


class PoissonStratifiedHMM(PoissonHMM):
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
            - ufl forms containing the cell problem and local stiffness assembly
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
        self._grad_v_macro = fem.Expression(
            ufl.grad(self._v_macro), self._reference_evaluation_point
        )
        # setup of placeholder functions for the micro functions
        self._v_micros = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
        # placeholders for the correctors
        self._correctors = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
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
                for i in range(self._num_basis_functions)
            ]
            for j in range(self._num_basis_functions)
        ]


class LinearElasticityHMM:
    r"""Solver for the Multi-Scale Linear Elasticity problem using the HMM.

    This class implements the Heterogenous-Multi-Scale Method for a Poisson problem.
    We want to solve the weak formulation of the Linear elasticity problem:

    Note that in this case $A = A_{ijkl}$ is a fourth order tensor and $e(u)$ is the strain of u
    and a matrix, i.e. $e(u)_{ij} = 1/2 (\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_j}).
    We define $Am = (A_{ijkl}m_{kl})_{ij}$

    $$
        \int_\Omega (A e(u)) : e(v) dx = \int_\Omega f \cdot v dx
    $$

    Note that we do not impose any Boundary condition by default.
    They have to be set by the user using [`set_boundary_condtions`][hommx.hmm.LinearElasticityHMM.set_boundary_condtions]

    We do this by approximating the homogenized coefficient on every cell and
    using the adapted bilinear form

    $$
    a_H(v_H, w_H) = \sum_{T\in \mathcal T_H} \frac{|T|}{|Y_\varepsilon(c_T)|} \int_{Y_\varepsilon(c_T)} (A(c_T, \frac{x}{\varepsilon}) (e(v_h) + e(\tilde{v_h})) : (e(v_h) + e(\tilde{v_h}) dx,
    $$

    where $\tilde{v_h}$ is the solution to

    $$
    \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon}) e(\tilde{v_h}) : e(z_h) dx = - \int_{Y_\varepsilon(c_T)} A(c_T, \frac{x}{\varepsilon})  e(v_H) : e(z_h) dx
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
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_LinearElasticityHMM",
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
        self._tdim = self._msh.topology.dim
        if self._tdim not in (2, 3):
            raise ValueError("Topology should be 3D or 2D")
        if self._tdim != self._msh.geometry.dim:
            raise ValueError(
                "Topological dimension is different from geometrical dimension. Currently surfaces in 3D are not supported."
            )
        if msh_micro.topology.dim != msh_micro.geometry.dim:
            raise ValueError(
                "Topological dimension is different from geometrical dimension for micro mesh."
            )
        if self._tdim != msh_micro.topology.dim:
            raise ValueError("Micro and macro mesh should have the same dimensionality.")
        self._comm = msh.comm
        self._coeff = A
        self._f = f
        self._eps = eps
        self._cell_mesh = msh_micro  # create a copy of the mesh, just in case
        self._V_macro = fem.functionspace(
            self._msh, ("Lagrange", 1, (self._tdim,))
        )  # Macroscopic function space
        self._v_test = ufl.TestFunction(self._V_macro)
        self._x = ufl.SpatialCoordinate(self._msh)
        L = ufl.inner(f(self._x), self._v_test) * ufl.dx
        self._L = fem.form(L)
        self._b = create_vector(self._L)
        self._u = fem.Function(self._V_macro)
        self._x = la.create_petsc_vector_wrap(self._u.x)
        self._bs = self._V_macro.dofmap.index_map_bs
        assert self._bs == self._tdim, (
            "block size and topological dimension should be equal. Please open an issue if you see this."
        )
        self._bcs = None

        # setup 2D vs 3D differences
        if self._tdim == 3:
            self._reference_evaluation_point = REFERENCE_EVALUATION_POINT_3D
            self._volume_function = _tetrahedron_volume

        if self._tdim == 2:
            self._reference_evaluation_point = REFERENCE_EVALUATION_POINT_2D
            self._volume_function = _triangle_area

        self._num_basis_functions = (
            self._tdim + 1
        ) * self._bs  # 3 basis functions for triangles, 4 for tetrahedra times block size
        self._num_dofs = self._V_macro.dofmap.index_map.size_global * self._bs
        self._num_local_dofs = self._V_macro.dofmap.index_map.size_local * self._bs
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

    @property
    def function_space(self) -> fem.FunctionSpace:
        """Function space of the macro mesh, that can be used to set Dirichlet BCs."""
        return self._V_macro

    def set_boundary_condtions(self, bcs: list[fem.DirichletBC] | fem.DirichletBC):
        """Set new boundary conditions. This method needs to be called before solve to accomodate
        the new boundary conditions, by default Dirichlet 0 on the whole boundary is enforced
        for squares or cubes.
        You can extract

        This method needs to be called if the geometry is not a square or cube.
        Args:
            bcs (list[fem.DirichletBC]): list of boundary conditions
        """
        if isinstance(bcs, list):
            self._bcs = bcs
        else:
            self._bcs = [bcs]

    def _assemble_stiffness(self):
        """Assembly of the stiffness matrix in parallel."""
        if self._A.assembled:
            return

        self._setup_cell_problems()
        num_local_cells = self._V_macro.mesh.topology.index_map(self._tdim).size_local

        # cell problem loop
        for local_cell_index in tqdm(range(num_local_cells)):
            local_dofs = self._V_macro.dofmap.cell_dofs(local_cell_index)
            global_dofs = self._V_macro.dofmap.index_map.local_to_global(local_dofs).astype(
                PETSc.IntType
            )
            # actual matrix entries are unrolled
            global_dofs_unrolled = _unroll_dofs(global_dofs, self._bs)
            # local assembly for one cell
            S_loc = self._compute_local_stiffness(local_cell_index)
            if np.any(np.isnan(S_loc)):
                self._logger.error(
                    f"Something went wrong when calculating local matrix on cell {local_cell_index}"
                )
            # global assembly
            self._A.setValues(
                global_dofs_unrolled,
                global_dofs_unrolled,
                S_loc.flatten(),
                addv=PETSc.InsertMode.ADD_VALUES,
            )

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
            - ufl forms containing the cell problem and local stiffness assembly
                - corresponding constants
        """
        # micro function space and periodic boundary conditions
        self._V_micro = fem.functionspace(self._cell_mesh, ("Lagrange", 1, (self._tdim,)))
        self._mpc = helpers.create_periodic_boundary_conditions(self._V_micro, self._bcs)
        self._points_micro = self._V_micro.tabulate_dof_coordinates()
        self._v_tilde = ufl.TrialFunction(self._V_micro)
        self._z = ufl.TestFunction(self._V_micro)
        self._y = ufl.SpatialCoordinate(self._cell_mesh)
        self._grad_v_micro = fem.Constant(self._cell_mesh, np.zeros((self._tdim, self._bs)))

        def e(u):
            return 1 / 2 * (ufl.grad(u) + ufl.transpose(ufl.grad(u)))

        # wrap x in A(x, y) in a Constant to avoid recompilation
        self._x_macro = fem.Constant(
            self._cell_mesh, np.zeros((self._cell_mesh.geometry.x.shape[1],))
        )
        self._A_micro = self._coeff(self._x_macro, self._y)
        i, j, k, l = ufl.indices(4)
        # precompile cell problem LHS and RHS
        self._a_micro_compiled = fem.form(
            ((self._A_micro)[i, j, k, l] * e(self._v_tilde)[k, l] * e(self._z)[i, j]) * ufl.dx
        )
        self._L_micro_compiled = fem.form(
            -((self._A_micro)[i, j, k, l] * self._grad_v_micro[k, l] * e(self._z)[i, j]) * ufl.dx
        )
        # setup of macro functions once for all cell problems
        self._v_macro = fem.Function(self._V_macro)
        self._strain_v_macro = fem.Expression(e(self._v_macro), self._reference_evaluation_point)
        # setup of placeholder functions for the micro functions
        self._v_micros = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
        # placeholders for the correctors
        self._correctors = [fem.Function(self._V_micro) for _ in range(self._num_basis_functions)]
        self._local_stiffness_forms = [
            [
                fem.form(
                    (
                        self._A_micro[i, j, k, l]
                        * (e(self._v_micros[i_loop])[k, l] + e(self._correctors[i_loop])[k, l])
                        * (e(self._v_micros[j_loop])[i, j] + e(self._correctors[j_loop])[i, j])
                    )
                    * ufl.dx
                )
                for i_loop in range(self._num_basis_functions)
            ]
            for j_loop in range(self._num_basis_functions)
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
        local_dofs_unrolled = _unroll_dofs(local_dofs, self._bs)
        points = self._V_macro.tabulate_dof_coordinates()[local_dofs]
        c_t = np.mean(points, axis=0)
        # update the x value in the precompiled forms containing A
        self._x_macro.value = c_t

        for i, local_dof in enumerate(local_dofs_unrolled):
            self._v_macro.x.array[:] = 0.0
            self._v_macro.x.array[local_dof] = 1.0
            self._interpolate_macro_to_micro(
                self._v_macro, self._strain_v_macro, cell_index, self._v_micros[i]
            )
            self._calculate_corrector(
                cell_index,
                self._correctors[i],
            )

        # local stiffness matrix
        S_loc = np.zeros((self._num_basis_functions, self._num_basis_functions))
        for i in range(S_loc.shape[0]):
            for j in range(S_loc.shape[1]):
                S_loc[i, j] = fem.assemble_scalar(self._local_stiffness_forms[i][j])

        # scale contribution
        cell_area = self._volume_function(points)
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
        self._grad_v_micro.value = grad_v_macro_expr.eval(self._msh, [cell_index]).reshape(
            (self._tdim, self._tdim)
        )
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
        if self._bcs is None:
            raise ValueError("You need to set Dirichlet Boundary conditions first.")
        # assemble LHS matrix
        self._assemble_stiffness()
        self._A.assemble()

        # assemble rhs
        with self._b.localForm() as b_local:
            b_local.set(0)
        with self._b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, self._L, None, None)
        self._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )  # accumulate ghost values on owning process
        self._b.assemble()

        # enforce Dirichlet BC by lifting
        for bc in self._bcs:
            local_bc_dofs, ghost_index = bc.dof_indices()  # process local
            global_bc_dofs = _local_to_global_unrolled(
                local_bc_dofs[:ghost_index], self._V_macro
            ).astype(PETSc.IntType)
            u_bc = self._u.copy()
            u_bc.x.array[:] = 0.0
            u_bc.x.array[local_bc_dofs] = np.tile(
                bc.g.value, local_bc_dofs.shape[0] // bc.g.value.shape[0]
            )
            u_bc.x.scatter_forward()
            b_lift = self._A.createVecLeft()
            self._A.mult(u_bc.x.petsc_vec, b_lift)

            bc.set(self._b)
            self._b.axpy(-1, b_lift)
            self._A.zeroRowsColumns(global_bc_dofs, diag=1.0)
            self._b.setValues(
                global_bc_dofs, np.tile(bc.g.value, local_bc_dofs.shape[0] // bc.g.value.shape[0])
            )
            self._b.assemble()

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


def _triangle_area(points):
    return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))


def _tetrahedron_volume(points):
    return (
        np.abs(np.linalg.det([points[1] - points[0], points[2] - points[0], points[3] - points[0]]))
        / 6.0
    )


def _unroll_dofs(dofs: np.ndarray, bs: int, dtype=PETSc.IntType):
    """unrolls blocked dofs into array indices"""
    dofs = np.asarray(dofs)
    assert len(dofs.shape) == 1, "Only flattened dof arrays allowed"
    ret = np.array(
        [[dofs[i] * bs + k for k in range(bs)] for i in range(len(dofs))], dtype=dtype
    ).flatten()
    return ret


def _local_to_global_unrolled(dofs: np.ndarray, V: fem.FunctionSpace):
    """maps local to global dofs, for unrolled dofs."""
    local_dofs_rerolled = dofs // V.dofmap.index_map_bs
    offsets = dofs % V.dofmap.index_map_bs
    global_dofs_unrolled = (
        V.dofmap.index_map.local_to_global(local_dofs_rerolled) * V.dofmap.index_map_bs + offsets
    )
    return global_dofs_unrolled
