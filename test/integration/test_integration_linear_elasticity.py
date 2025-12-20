import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from hommx.hmm import LinearElasticityHMM
from hommx.petsc_helper import petsc_matrix_to_numpy


@pytest.fixture
def atol_2d():
    return 0.05


@pytest.fixture
def eps_2d():
    return 1 / 2**3


@pytest.fixture
def beam_width():
    return 0.2


@pytest.fixture
def beam_length():
    return 1.0


@pytest.fixture
def macro_mesh_2d(beam_width, beam_length):
    return mesh.create_rectangle(
        MPI.COMM_SELF,
        [np.array([0, 0]), np.array([beam_length, beam_width])],
        [40, 12],
        cell_type=mesh.CellType.triangle,
    )


@pytest.fixture
def micro_mesh_2d():
    return mesh.create_unit_square(
        MPI.COMM_SELF,
        10,
        10,
        cell_type=mesh.CellType.triangle,
    )


@pytest.fixture
def reference_mesh_2d(beam_width, beam_length):
    return mesh.create_rectangle(
        MPI.COMM_SELF,
        [np.array([0, 0]), np.array([beam_length, beam_width])],
        [800, 240],
        cell_type=mesh.CellType.triangle,
    )


def test_linear_elasticity_2d(
    micro_mesh_2d, macro_mesh_2d, reference_mesh_2d, eps_2d, atol_2d, beam_length, beam_width
):
    """Test FE-HMM on linear elasticity heuristically by solving on a rather fine mesh using FEM
    and comparing against the FE-HMM."""
    rho = 1.0
    delta = beam_width / beam_length
    gamma = 0.4 * delta**2
    g = gamma
    dim = 2
    fdim = dim - 1  # boundary facet dimension
    u_D = np.array([0, 0], dtype=default_scalar_type)  # Dirichlet BC

    def epsilon(u):
        return 1 / 2 * (ufl.grad(u) + ufl.transpose(ufl.grad(u)))

    def mu(x, y):
        return 5 + 4.5 * ufl.sin(2 * ufl.pi * y[0])

    def lambda_(x, y):
        return 1.25

    def A_tensor(x, y):
        """Elasticity tensor A such that A:ε(u) gives stress."""
        I = ufl.Identity(dim)
        i, j, k, l = ufl.indices(4)

        A = ufl.as_tensor(
            lambda_(x, y) * I[i, j] * I[k, l] + mu(x, y) * (I[i, k] * I[j, l] + I[i, l] * I[j, k]),
            indices=(i, j, k, l),
        )
        return A

    COMM = MPI.COMM_WORLD

    f_reference = fem.Constant(reference_mesh_2d, default_scalar_type((0, -rho * g)))

    def f_hmm(
        x,
    ):  # this supresses some warnings because we call f(x) inside hmm code (general source term)
        return fem.Constant(macro_mesh_2d, default_scalar_type((0, -rho * g)))

    # FEM setup
    i, j, k, l = ufl.indices(4)
    V_ref = fem.functionspace(reference_mesh_2d, ("Lagrange", 1, (2,)))  # hard code 2D
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    x = ufl.SpatialCoordinate(reference_mesh_2d)
    lhs = (A_tensor(x, x / eps_2d)[i, j, k, l] * epsilon(u)[k, l] * epsilon(v)[i, j]) * ufl.dx
    rhs = ufl.dot(f_reference, v) * ufl.dx

    # BC
    def clamped_boundary(x):
        return np.isclose(x[0], 0)

    boundary_facets = mesh.locate_entities_boundary(reference_mesh_2d, fdim, clamped_boundary)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V_ref, fdim, boundary_facets), V_ref)

    bcs = [bc]
    lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    u_ref = lp.solve()

    lehmm = LinearElasticityHMM(
        macro_mesh_2d,
        A_tensor,
        f_hmm,
        micro_mesh_2d,
        eps_2d,
        petsc_options_cell_problem={"ksp_atol": 1e-9},
    )

    boundary_facets = mesh.locate_entities_boundary(macro_mesh_2d, fdim, clamped_boundary)
    bc = fem.dirichletbc(
        u_D,
        fem.locate_dofs_topological(lehmm.function_space, fdim, boundary_facets),
        lehmm.function_space,
    )
    lehmm.set_boundary_conditions(bc)

    u_lehmm = lehmm.solve()

    def interpolate_nonmatching(
        V_to: fem.FunctionSpace, V_from: fem.FunctionSpace, func: fem.Function
    ) -> fem.Function:
        tdim = V_to.mesh.topology.dim
        cells_V_to = np.arange(V_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        interpolation_data = fem.create_interpolation_data(V_to, V_from, cells_V_to)
        func_V_to = fem.Function(V_to)
        func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
        func_V_to.x.scatter_forward()
        return func_V_to

    def calc_l2_error(u1, u2):
        return np.sqrt(
            COMM.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(u1 - u2, u1 - u2) * ufl.dx)), op=MPI.SUM
            )
        )

    def calc_l2_norm(u1):
        return np.sqrt(
            COMM.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u1, u1) * ufl.dx)), op=MPI.SUM)
        )

    u_ref_interpolated = interpolate_nonmatching(u_lehmm._V, V_ref, u_ref)
    relative_error = calc_l2_error(u_lehmm, u_ref_interpolated) / calc_l2_norm(u_ref_interpolated)

    assert relative_error < atol_2d, (
        f"Relative error in 3D HMM too high. This is a heuristic, to check for code regressions. {relative_error=}"
    )


@pytest.fixture
def atol_3d():
    return 1e-4


@pytest.fixture
def eps_3d():
    return 1  # not used anyways


@pytest.fixture
def macro_mesh_3d(beam_width, beam_length):
    return mesh.create_box(
        MPI.COMM_SELF,
        [np.array([0, 0, 0]), np.array([beam_length, beam_width, beam_width])],
        [10, 3, 3],
        cell_type=mesh.CellType.tetrahedron,
    )


@pytest.fixture
def micro_mesh_3d():
    return mesh.create_unit_cube(
        MPI.COMM_SELF,
        3,
        3,
        3,
        cell_type=mesh.CellType.tetrahedron,
    )


def test_linear_elasticity_3d(
    micro_mesh_3d, macro_mesh_3d, eps_3d, atol_3d, beam_length, beam_width
):
    """Since a fine FEM mesh is hard to do, we instead just compare them with no periodic
    component at all, this is rather a test if 3D linear elasticity runs through."""
    rho = 1.0
    delta = beam_width / beam_length
    gamma = 0.4 * delta**2
    g = gamma
    dim = 3
    fdim = dim - 1  # boundary facet dimension
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)  # Dirichlet BC

    def epsilon(u):
        return 1 / 2 * (ufl.grad(u) + ufl.transpose(ufl.grad(u)))

    def mu(x, y):
        return 1

    def lambda_(x, y):
        return 1.25

    def A_tensor(x, y):
        """Elasticity tensor A such that A:ε(u) gives stress."""
        I = ufl.Identity(dim)
        i, j, k, l = ufl.indices(4)

        A = ufl.as_tensor(
            lambda_(x, y) * I[i, j] * I[k, l] + mu(x, y) * (I[i, k] * I[j, l] + I[i, l] * I[j, k]),
            indices=(i, j, k, l),
        )
        return A

    COMM = MPI.COMM_SELF

    f_reference = fem.Constant(macro_mesh_3d, default_scalar_type((0, 0, -rho * g)))

    def f_hmm(
        x,
    ):  # this supresses some warnings because we call f(x) inside hmm code (general source term)
        return fem.Constant(macro_mesh_3d, default_scalar_type((0, 0, -rho * g)))

    # FEM setup
    i, j, k, l = ufl.indices(4)
    V_ref = fem.functionspace(macro_mesh_3d, ("Lagrange", 1, (dim,)))  # hard code 2D
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    x = ufl.SpatialCoordinate(macro_mesh_3d)
    lhs = (A_tensor(x, x / eps_3d)[i, j, k, l] * epsilon(u)[k, l] * epsilon(v)[i, j]) * ufl.dx
    rhs = ufl.dot(f_reference, v) * ufl.dx

    # BC
    def clamped_boundary(x):
        return np.isclose(x[0], 0)

    boundary_facets = mesh.locate_entities_boundary(macro_mesh_3d, fdim, clamped_boundary)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V_ref, fdim, boundary_facets), V_ref)

    bcs = [bc]
    lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    u_ref = lp.solve()

    lehmm = LinearElasticityHMM(
        macro_mesh_3d,
        A_tensor,
        f_hmm,
        micro_mesh_3d,
        eps_3d,
        petsc_options_cell_problem={"ksp_atol": 1e-9},
    )

    boundary_facets = mesh.locate_entities_boundary(macro_mesh_3d, fdim, clamped_boundary)
    bc = fem.dirichletbc(
        u_D,
        fem.locate_dofs_topological(lehmm.function_space, fdim, boundary_facets),
        lehmm.function_space,
    )
    lehmm.set_boundary_conditions(bc)

    u_lehmm = lehmm.solve()

    def interpolate_nonmatching(
        V_to: fem.FunctionSpace, V_from: fem.FunctionSpace, func: fem.Function
    ) -> fem.Function:
        tdim = V_to.mesh.topology.dim
        cells_V_to = np.arange(V_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        interpolation_data = fem.create_interpolation_data(V_to, V_from, cells_V_to)
        func_V_to = fem.Function(V_to)
        func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
        func_V_to.x.scatter_forward()
        return func_V_to

    def calc_l2_error(u1, u2):
        return np.sqrt(
            COMM.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(u1 - u2, u1 - u2) * ufl.dx)), op=MPI.SUM
            )
        )

    def calc_l2_norm(u1):
        return np.sqrt(
            COMM.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u1, u1) * ufl.dx)), op=MPI.SUM)
        )

    u_ref_interpolated = interpolate_nonmatching(u_lehmm._V, V_ref, u_ref)
    relative_error = calc_l2_error(u_lehmm, u_ref_interpolated) / calc_l2_norm(u_ref_interpolated)

    A_fem = petsc_matrix_to_numpy(lp.A)
    A_lehmm = petsc_matrix_to_numpy(lehmm._A)
    matrix_relative_error = np.linalg.norm(A_fem - A_lehmm) / np.linalg.norm(A_fem)

    assert matrix_relative_error < atol_3d, (
        f"Relative error in between matrices is to large. {matrix_relative_error=}"
    )

    assert relative_error < atol_3d, (
        f"Relative error in 3D HMM too high. This is a heuristic, to check for code regressions. {relative_error=}"
    )
