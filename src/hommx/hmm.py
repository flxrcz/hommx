"""Base class for HMM solvers using DOLFINx."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, la, mesh, plot
from dolfinx.fem.assemble import _assemble_vector_array
from dolfinx.fem.petsc import create_vector
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

import hommx.cell_problem as cell_problem
import hommx.helpers as helpers


def _triangle_area(points):
    return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))


def _tetrahedron_volume(points):
    return (
        np.abs(np.linalg.det([points[1] - points[0], points[2] - points[0], points[3] - points[0]]))
        / 6.0
    )


def _unroll_dofs(dofs: np.ndarray, bs: int, dtype=PETSc.IntType):
    """unrolls blocked dofs into array indices"""
    if bs == 1:  # non-blocked dofs don't need to be unrolled
        return dofs
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


class BaseHMM(ABC):
    r"""Abstract base class for Heterogeneous Multi-Scale Method solvers.

    Provides common infrastructure for scalar and vector-valued HMM problems.
    Subclasses must implement:
    - `_setup_macro_function_space()`: Set up macro-scale function space
    - `_setup_micro_function_space()`: Set up micro-scale function space
    - `_setup_cell_problem_forms()`: Set up micro-scale cell problem forms, take a look at PoissonHMM for an example
    """

    def __init__(
        self,
        msh: mesh.Mesh,
        A: Callable[[fem.Constant, ufl.SpatialCoordinate], ufl.Form],
        f: Callable[[ufl.SpatialCoordinate], ufl.Form],
        msh_micro: mesh.Mesh,
        eps: float,
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_HMM",
    ):
        r"""Initializes the solver, meshes and boundary conditions.

        Args:
            msh: The macro mesh, on which we want to solve the oscillatory PDE.
            A: The coefficient, should be callable like: `A(x)(y)`,
                where x is a spatial coordinate on the macro mesh (the cell center c_T)
                and y is a ufl.SpatialCoordinate on the microscopic mesh.
                A needs to be 1-periodic in y.
            f: The right hand side of the problem.
            msh_micro: The microscopic mesh, needs to be the unit cell.
                Should live on MPI.COMM_SELF since each process owns a whole copy.
            eps: The microscopic scaling parameter.
            petsc_options_global_solve: PETSc solver options for the global solver.
            petsc_options_cell_problem: PETSc solver options for the cell problem solver.
            petsc_options_prefix: Options prefix used for PETSc options.
        """
        self._logger = logging.getLogger(__name__)
        self._msh = msh
        self._comm = msh.comm
        self._coeff = A
        self._f = f
        self._eps = eps
        self._cell_mesh = msh_micro
        if self._cell_mesh.comm.Compare(MPI.COMM_SELF) != 1:
            self._logger.error(
                "Cell mesh shold be on MPI.COMM_SELF, other communicators may not work."
            )
        self._cell_mesh_area = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=self._cell_mesh)))
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

        # Setup dimension-dependent functions
        if self._tdim == 3:
            self._volume_function = _tetrahedron_volume
        else:
            self._volume_function = _triangle_area

        # Setup function space (subclass-specific)
        self._V_macro = self._setup_macro_function_space()
        self._bs = self._V_macro.dofmap.index_map_bs

        # Setup RHS and forms
        self._v_test = ufl.TestFunction(self._V_macro)
        self._x = ufl.SpatialCoordinate(self._msh)
        L = ufl.inner(f(self._x), self._v_test) * ufl.dx
        self._L = fem.form(L)
        self._b = create_vector(self._L)
        self._u = fem.Function(self._V_macro)
        self._x = la.create_petsc_vector_wrap(self._u.x)

        # Setup matrix dimensions
        self._num_basis_functions_per_cell = (
            self._V_macro.dofmap.dof_layout.num_dofs
        ) * self._bs  # 3 basis functions for triangles, 4 for tetrahedra times block size
        self._num_global_dofs = self._V_macro.dofmap.index_map.size_global * self._bs
        self._num_local_dofs = self._V_macro.dofmap.index_map.size_local * self._bs

        self._A = PETSc.Mat().createAIJ(
            (
                (self._num_local_dofs, self._num_global_dofs),
                (self._num_local_dofs, self._num_global_dofs),
            )
        )
        self._needs_reassembly = True

        # Setup solver
        if petsc_options_cell_problem is None:
            petsc_options_cell_problem = {"ksp_atol": 1e-10}  # better default
        self._petsc_options_cell_problem = petsc_options_cell_problem

        self._solver = PETSc.KSP().create(self._comm)
        self._solver.setOptionsPrefix(petsc_options_prefix)
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(petsc_options_prefix)
        if petsc_options_global_solve is not None:
            for k, v in petsc_options_global_solve.items():
                opts[k] = v
            self._solver.setFromOptions()
            for k in petsc_options_global_solve.keys():
                del opts[k]

        opts.prefixPop()
        self._bcs = []

        self._setup_cell_problem_variables()

    @property
    def function_space(self) -> fem.FunctionSpace:
        """Function space of the macro mesh."""
        return self._V_macro

    def _setup_cell_problem_variables(self) -> None:
        """Setup cell problem specifics"""
        # micro function space and periodic boundary conditions
        self._V_micro = self._setup_micro_function_space()
        self._mpc = helpers.create_periodic_boundary_conditions(self._V_micro, self._bcs)
        self._points_micro = self._V_micro.tabulate_dof_coordinates()
        self._v_tilde = ufl.TrialFunction(self._V_micro)
        self._z = ufl.TestFunction(self._V_micro)
        self._y = ufl.SpatialCoordinate(self._cell_mesh)

        # setup constants that are reused throughout
        # constant that can be used in the cell problem for the slow variable
        self._x_macro = fem.Constant(
            self._cell_mesh, np.zeros((self._cell_mesh.geometry.x.shape[1],))
        )

        # placeholder function that can be used in the cell problems
        self._v_macro = fem.Function(self._V_macro)

        # coefficient
        self._A_micro = self._coeff(self._x_macro, self._y)

        # setup of placeholder functions for the micro functions
        self._v_micros = [
            fem.Function(self._V_micro) for _ in range(self._num_basis_functions_per_cell)
        ]
        # placeholders for the correctors
        self._correctors = [
            fem.Function(self._V_micro) for _ in range(self._num_basis_functions_per_cell)
        ]

    @abstractmethod
    def _setup_macro_function_space(self) -> fem.FunctionSpace:
        """Setup macro-scale function space. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _setup_micro_function_space(self) -> fem.FunctionSpace:
        """Setup micro-scale function space. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_cell_problem_lhs(self) -> fem.Form:
        """Builds the LHS of the cell problem. Must be implemented by subclasses
            Please use the variables in _setup_cell_problem_variables inside the forms for efficiency.
        Returns:
            fem.Form: lhs form for the cell problem
        """
        pass

    @abstractmethod
    def _build_cell_problem_rhs(self, v_micro) -> fem.Form:
        """Builds the LHS of the cell problem. Must be implemented by subclasses
            Please use the variables in _setup_cell_problem_variables inside the forms for efficiency.
        Args:
            v_micro: Macroscopic basis function interpolated onto the micro mesh
        Returns:
            fem.Form: rhs form for the cell problem
        """
        pass

    @abstractmethod
    def _build_local_stiffness_contribution(
        self,
        v_micro_i: fem.Function,
        v_micro_j: fem.Function,
        corrector_i: fem.Function,
        corrector_j: fem.Function,
    ):
        """Builds the form corresponding to the local stiffness contribution $S_{ij}$ for one element. Must be implemented by subclasses
            Please use the variables in _setup_cell_problem_variables inside the forms for efficiency.

        v_micro_i: i-th basis function on the element
        v_micro_i: j-th basis function on the element
        corrector_i: corrector corresponding to v_micro_i
        corrector_j: corrector corresponding to v_micro_j
        Returns:
            fem.Form: lhs form for the cell problem
        """
        pass

    def _setup_cell_problem_forms(self) -> None:
        """Setup cell problem forms, from abstract base class method."""
        self._a_micro_compiled = self._build_cell_problem_lhs()
        self._L_micro_compiled = [
            self._build_cell_problem_rhs(self._v_micros[i])
            for i in range(self._num_basis_functions_per_cell)
        ]
        self._local_stiffness_forms = [
            [
                self._build_local_stiffness_contribution(
                    self._v_micros[i], self._v_micros[j], self._correctors[i], self._correctors[j]
                )
                for i in range(self._num_basis_functions_per_cell)
            ]
            for j in range(self._num_basis_functions_per_cell)
        ]

    def set_boundary_conditions(self, bcs: list[fem.DirichletBC] | fem.DirichletBC):
        """Set boundary conditions.

        Args:
            bcs: Single BC or list of BCs
        """
        if isinstance(bcs, list):
            self._bcs = bcs
        else:
            self._bcs = [bcs]
        # Mark for reassembly; we keep the preallocation and just zero on next assembly
        self._needs_reassembly = True

    def set_right_hand_side(self, f: Callable[[ufl.SpatialCoordinate], ufl.Form]):
        """Sets the right hand side

        Args:
            f: The right hand side of the problem.
        """
        L = ufl.inner(f(self._x), self._v_test) * ufl.dx
        self._L = fem.form(L)

    def _assemble_stiffness(self):
        """Assembly of the stiffness matrix in parallel."""
        if not self._needs_reassembly:
            return

        # Reuse preallocation; just clear existing values
        self._A.zeroEntries()

        self._setup_cell_problem_forms()
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

        self._needs_reassembly = False

    def _compute_local_stiffness(self, cell_index: int) -> np.ndarray:
        """Computes the local stiffness matrix on one element by solving the cell problem for all
        basis functions on that cell.

        Args:
            cell_index: process-local index of the cell for which the homogenized coefficient
            is to be approximated

        Returns:
            np.ndarray: local stiffness matrix
        """
        local_dofs = self._V_macro.dofmap.cell_dofs(cell_index)
        local_dofs_unrolled = _unroll_dofs(local_dofs, self._bs)
        # Ensure connectivity between cells and vertices is available
        self._msh.topology.create_connectivity(self._tdim, 0)
        cell_vertices = self._msh.geometry.x[
            self._msh.topology.connectivity(self._tdim, 0).links(cell_index)
        ]
        c_t = np.mean(cell_vertices, axis=0)
        # update the x value in the precompiled forms containing A
        self._x_macro.value = c_t

        for i, local_dof in enumerate(local_dofs_unrolled):
            self._v_macro.x.array[:] = 0.0
            self._v_macro.x.array[local_dof] = 1.0
            self._interpolate_macro_to_micro(self._v_macro, self._v_micros[i], cell_index)
            self._calculate_corrector(cell_index, i, self._correctors[i])

        # local stiffness matrix
        S_loc = np.zeros((self._num_basis_functions_per_cell, self._num_basis_functions_per_cell))
        for i in range(S_loc.shape[0]):
            for j in range(S_loc.shape[1]):
                S_loc[i, j] = fem.assemble_scalar(self._local_stiffness_forms[i][j])

        # scale contribution
        cell_area = self._volume_function(cell_vertices)
        Y_area = self._cell_mesh_area
        return S_loc * cell_area / Y_area

    def _interpolate_macro_to_micro(
        self,
        v_macro: fem.Function,
        v_micro: fem.Function,
        cell_index: int,
    ) -> fem.Function:
        """Interpolates a function from the macro mesh onto the micro mesh.

        Since we know that the micro domain is contained inside one macro cell,
        we can avoid dolfinx interpolate_nonmatching and instead rely on evaluating directly.

        Args:
            v_macro: Macro function
            cell_index: cell index on which the cell problem is solved
            v_micro: Micro function, if none is provided one is created

        """
        cells = np.full(self._points_micro.shape[0], cell_index, dtype=np.int32)
        c_t = self._x_macro.value
        micro_center = np.mean(self._points_micro, axis=0)
        v_micro.x.array[:] = v_macro.eval(
            (self._points_micro - micro_center) * self._eps + c_t, cells=cells
        ).flatten()

        return v_micro

    def _calculate_corrector(
        self,
        cell_index: int,
        cell_problem_index: int,
        corrector: fem.Function | None = None,
    ):
        """Calculates the corrector by solving the cell problem.

        Notes:
            The micro function for which the corrector should be calculated does not show up,
            since the precompiled form uses a constant that is updated in _interpolate_macro_to_micro.

        Args:
            cell_index: the process-local index of the cell
                on which the cell problem should be solved
            cell_problem_index: local index of the basis function for which the corrector should be calculated.
                By default this is used to determine which RHS to use, so that the right basis function is used on the rhs.
            corrector: function on the micro mesh that stores the corrector,
                if none is provided one is created
        """
        if corrector is None:
            corrector = fem.Function(self._V_micro)

        problem = cell_problem.PeriodicLinearProblem(
            self._a_micro_compiled,
            self._L_micro_compiled[cell_problem_index],
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
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self._b.assemble()

        # enforce Dirichlet BC by lifting
        for bc in self._bcs:
            local_bc_dofs, ghost_index = bc.dof_indices()
            owned_local_bc_dofs = local_bc_dofs[:ghost_index]
            global_bc_dofs = _local_to_global_unrolled(owned_local_bc_dofs, self._V_macro).astype(
                PETSc.IntType
            )
            if hasattr(bc.g, "x"):  # function valued bc
                bc_value = bc.g.x.array[owned_local_bc_dofs]
            elif bc.g.value.ndim == 0:  # scalar valued bc
                bc_value = np.full(owned_local_bc_dofs.shape, bc.g.value)
            else:  # vector valued bc
                bc_value = np.tile(
                    bc.g.value,
                    owned_local_bc_dofs.shape[0] // bc.g.value.shape[0],
                )

            u_bc = self._u.copy()
            u_bc.x.array[:] = 0.0
            u_bc.x.array[owned_local_bc_dofs] = bc_value
            u_bc.x.scatter_forward()
            b_lift = self._A.createVecLeft()
            self._A.mult(u_bc.x.petsc_vec, b_lift)

            bc.set(self._b)
            self._b.axpy(-1, b_lift)
            self._A.zeroRowsColumns(global_bc_dofs, diag=1.0)
            self._b.setValues(global_bc_dofs, bc_value)
            self._b.assemble()

        self._solver.setOperators(self._A)
        self._solver.solve(self._b, self._x)

        if self._solver.getConvergedReason() < 0:
            self._logger.error(
                f"Something went wrong in the global problem solve. PETSc solver failed with reason {self._solver.getConvergedReason()}"
            )

        self._u.x.scatter_forward()
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


class PoissonHMM(BaseHMM):
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
        f: Callable,
        msh_micro: mesh.Mesh,
        eps: float,
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_PoissonHMM",
    ):
        r"""Initializes the solver, meshes and boundary conditions.

        Args:
            msh: The macro mesh, on which we want to solve the oscillatory PDE.
            A: The coefficient, should be callable like: `A(x)(y)`,
                where x is a spatial coordinate on the macro mesh (the cell center c_T)
                and y is a ufl.SpatialCoordinate on the microscopic mesh.
                A needs to be 1-periodic in y.
            f: The right hand side of the problem.
            msh_micro: The microscopic mesh, needs to be the unit cell.
                Should live on MPI.COMM_SELF since each process owns a whole copy.
            eps: The microscopic scaling parameter.
            petsc_options_global_solve: PETSc solver options for the global solver.
            petsc_options_cell_problem: PETSc solver options for the cell problem solver.
            petsc_options_prefix: Options prefix used for PETSc options.
        """
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

        dofs = fem.locate_dofs_topological(
            self._V_macro, entity_dim=(self._msh.topology.dim - 1), entities=facets
        )
        bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=self._V_macro)
        self._bcs = [bc]

    def _setup_macro_function_space(self) -> fem.FunctionSpace:
        return fem.functionspace(self._msh, ("Lagrange", 1))

    def _setup_micro_function_space(self) -> fem.FunctionSpace:
        return fem.functionspace(self._cell_mesh, ("Lagrange", 1))

    def _build_cell_problem_lhs(self) -> fem.Form:
        return fem.form(
            ufl.inner(self._A_micro * ufl.grad(self._v_tilde), ufl.grad(self._z)) * ufl.dx
        )

    def _build_cell_problem_rhs(self, v_micro: fem.Function) -> fem.Form:
        return fem.form(-ufl.inner(self._A_micro * ufl.grad(v_micro), ufl.grad(self._z)) * ufl.dx)

    def _build_local_stiffness_contribution(
        self,
        v_micro_i: fem.Function,
        v_micro_j: fem.Function,
        corrector_i: fem.Function,
        corrector_j: fem.Function,
    ):
        return fem.form(
            1
            / self._eps**2
            * ufl.inner(
                self._A_micro * (ufl.grad(v_micro_i) + ufl.grad(corrector_i)),
                ufl.grad(v_micro_j) + ufl.grad(corrector_j),
            )
            * ufl.dx
        )


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
        petsc_options_prefix: str = "hommx_PoissonStratifiedHMM",
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
        self._Dthetax = self._Dtheta(self._x_macro)

    def _build_cell_problem_lhs(self) -> fem.Form:
        return fem.form(
            ufl.inner(
                self._A_micro * self._Dthetax * ufl.grad(self._v_tilde),
                self._Dthetax * ufl.grad(self._z),
            )
            * ufl.dx
        )

    def _build_cell_problem_rhs(self, v_micro: fem.Function) -> fem.Form:
        return fem.form(
            -ufl.inner(self._A_micro * ufl.grad(v_micro), self._Dthetax * ufl.grad(self._z))
            * ufl.dx
        )

    def _build_local_stiffness_contribution(
        self,
        v_micro_i: fem.Function,
        v_micro_j: fem.Function,
        corrector_i: fem.Function,
        corrector_j: fem.Function,
    ):
        return fem.form(
            1
            / self._eps**2
            * ufl.inner(
                self._A_micro * (ufl.grad(v_micro_i) + self._Dthetax * ufl.grad(corrector_i)),
                (ufl.grad(v_micro_j) + self._Dthetax * ufl.grad(corrector_j)),
            )
            * ufl.dx
        )


class LinearElasticityHMM(BaseHMM):
    r"""Solver for the Multi-Scale Linear Elasticity problem using the HMM.

    This class implements the Heterogenous-Multi-Scale Method for a Linear elasticity problem.
    We want to solve the weak formulation of the Linear elasticity problem:

    Note that in this case $A = A_{ijkl}$ is a fourth order tensor and $e(u)$ is the strain of u
    and a matrix, i.e. $e(u)_{ij} = 1/2 (\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_j}).
    We define $Am = (A_{ijkl}m_{kl})_{ij}$

    $$
        \int_\Omega (A e(u)) : e(v) dx = \int_\Omega f \cdot v dx
    $$

    Note that we do not impose any Boundary condition by default.
    They have to be set by the user using [`set_boundary_conditions`][hommx.hmm.LinearElasticityHMM.set_boundary_conditions]

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
        f: Callable,
        msh_micro: mesh.Mesh,
        eps: float,
        petsc_options_global_solve: dict | None = None,
        petsc_options_cell_problem: dict | None = None,
        petsc_options_prefix: str = "hommx_LinearElasticityHMM",
    ):
        r"""Initializes the solver, meshes and boundary conditions.

        Args:
            msh: The macro mesh, on which we want to solve the oscillatory PDE.
            A: The coefficient, should be callable like: `A(x)(y)`,
                where x is a spatial coordinate on the macro mesh (the cell center c_T)
                and y is a ufl.SpatialCoordinate on the microscopic mesh.
                A needs to be 1-periodic in y.
            f: The right hand side of the problem.
            msh_micro: The microscopic mesh, needs to be the unit cell.
                Should live on MPI.COMM_SELF since each process owns a whole copy.
            eps: The microscopic scaling parameter.
            petsc_options_global_solve: PETSc solver options for the global solver.
            petsc_options_cell_problem: PETSc solver options for the cell problem solver.
            petsc_options_prefix: Options prefix used for PETSc options.
        """
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

    def _setup_macro_function_space(self) -> fem.FunctionSpace:
        return fem.functionspace(self._msh, ("Lagrange", 1, (self._tdim,)))

    def _setup_micro_function_space(self) -> fem.FunctionSpace:
        return fem.functionspace(self._cell_mesh, ("Lagrange", 1, (self._tdim,)))

    @staticmethod
    def _e(u):
        return 1 / 2 * (ufl.grad(u) + ufl.transpose(ufl.grad(u)))

    def _build_cell_problem_lhs(self) -> fem.Form:
        i, j, k, l = ufl.indices(4)
        return fem.form(
            ((self._A_micro)[i, j, k, l] * self._e(self._v_tilde)[k, l] * self._e(self._z)[i, j])
            * ufl.dx
        )

    def _build_cell_problem_rhs(self, v_micro: fem.Function) -> fem.Form:
        i, j, k, l = ufl.indices(4)
        return fem.form(
            -((self._A_micro)[i, j, k, l] * self._e(v_micro)[k, l] * self._e(self._z)[i, j])
            * ufl.dx
        )

    def _build_local_stiffness_contribution(
        self,
        v_micro_i: fem.Function,
        v_micro_j: fem.Function,
        corrector_i: fem.Function,
        corrector_j: fem.Function,
    ):
        i, j, k, l = ufl.indices(4)
        return fem.form(
            1
            / self._eps**2
            * (
                self._A_micro[i, j, k, l]
                * (self._e(v_micro_i)[k, l] + self._e(corrector_i)[k, l])
                * (self._e(v_micro_j)[i, j] + self._e(corrector_j)[i, j])
            )
            * ufl.dx
        )
