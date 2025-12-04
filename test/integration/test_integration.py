import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.hmm import PoissonHMM


@pytest.fixture
def mesh_sizes():
    return 15, 15


@pytest.fixture
def atol():
    return 5e-5


@pytest.fixture
def eps(mesh_sizes):
    discretization = 1 / min(mesh_sizes)
    return 0.1 * discretization


@pytest.fixture
def macro_mesh(mesh_sizes):
    return mesh.create_unit_square(MPI.COMM_SELF, *mesh_sizes)


@pytest.fixture
def micro_mesh(mesh_sizes):
    return mesh.create_unit_square(MPI.COMM_SELF, *mesh_sizes)


@pytest.fixture
def mesh_sizes_3d():
    return 6, 6, 6


@pytest.fixture
def mesh_reference_sizes_3d():
    return 2**6, 2**6, 2**6


@pytest.fixture
def atol_3d():
    return 0.05  # heuristic value


@pytest.fixture
def eps_3d():
    return 1 / 2**3


@pytest.fixture
def macro_mesh_3d(mesh_sizes_3d):
    return mesh.create_unit_cube(MPI.COMM_SELF, *mesh_sizes_3d)


@pytest.fixture
def micro_mesh_3d(mesh_sizes_3d):
    return mesh.create_unit_cube(MPI.COMM_SELF, *mesh_sizes_3d)


@pytest.fixture
def reference_mesh_3d(mesh_reference_sizes_3d):
    return mesh.create_unit_cube(MPI.COMM_SELF, *mesh_reference_sizes_3d)


def test_analytical_example_1(micro_mesh, macro_mesh, eps, atol):
    """Stolen from Felix Krumbiegels HMM code"""

    def A(x, y):
        return 1 / (2 + ufl.cos(2 * ufl.pi * y[0]))

    def f(x):
        return (
            ufl.pi**2 * (1 / 2 + 1 / ufl.sqrt(3)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        )

    def solution(x):
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    phmm = PoissonHMM(macro_mesh, A, f, micro_mesh, eps)
    phmm_solution = phmm.solve()
    u_exact = solution(ufl.SpatialCoordinate(macro_mesh))
    L2_error = fem.assemble_scalar(
        fem.form(ufl.inner(phmm_solution - u_exact, phmm_solution - u_exact) * ufl.dx)
    )
    assert np.isclose(L2_error, 0, atol=atol), f"L^2 error too big {L2_error=}"


def test_analytical_example_2(micro_mesh, macro_mesh, eps, atol):
    """Stolen from Felix Krumbiegels HMM code"""

    def A(x, y):
        return 0.33 + 0.15 * (ufl.sin(2 * ufl.pi * x[0]) + ufl.sin(2 * ufl.pi * y[0]))

    def f(x):
        return (
            3.25696945235949
            * ufl.sqrt(
                (0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1) ** 2 - 0.206611570247934
            )
            * ufl.sin(ufl.pi * x[0])
            * ufl.sin(ufl.pi * x[1])
            + ufl.pi**2
            * (0.15 * ufl.sin(2 * ufl.pi * x[0]) + 0.33)
            * ufl.sin(ufl.pi * x[0])
            * ufl.sin(ufl.pi * x[1])
            - 2.96088132032681
            * (0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1)
            * ufl.sin(ufl.pi * x[1])
            * ufl.cos(ufl.pi * x[0])
            * ufl.cos(2 * ufl.pi * x[0])
            / ufl.sqrt(
                (0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1) ** 2 - 0.206611570247934
            )
        )

    def solution(x):
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    phmm = PoissonHMM(macro_mesh, A, f, micro_mesh, eps)
    phmm_solution = phmm.solve()
    u_exact = solution(ufl.SpatialCoordinate(macro_mesh))
    L2_error = fem.assemble_scalar(
        fem.form(ufl.inner(phmm_solution - u_exact, phmm_solution - u_exact) * ufl.dx)
    )
    assert np.isclose(L2_error, 0, atol=atol), f"L^2 error too big {L2_error=}"


def test_3d(micro_mesh_3d, macro_mesh_3d, reference_mesh_3d, eps_3d, atol_3d):
    def A(x, y):
        return 1.1 + x[0] + ufl.sin(2 * ufl.pi * y[0])

    def f(x):
        return 1

    COMM = MPI.COMM_WORLD

    phmm = PoissonHMM(
        macro_mesh_3d, A, f, micro_mesh_3d, eps_3d, petsc_options_cell_problem={"ksp_atol": 1e-9}
    )

    def A_fem(x):
        return A(x, x / eps_3d)

    V_ref = fem.functionspace(reference_mesh_3d, ("Lagrange", 1))
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    x = ufl.SpatialCoordinate(reference_mesh_3d)
    lhs = ufl.inner(A_fem(x) * ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(f(x), v) * ufl.dx
    left = np.min(reference_mesh_3d.geometry.x[:, 0])
    right = np.max(reference_mesh_3d.geometry.x[:, 0])
    bottom = np.min(reference_mesh_3d.geometry.x[:, 1])
    top = np.max(reference_mesh_3d.geometry.x[:, 1])
    back = np.min(reference_mesh_3d.geometry.x[:, 2])
    front = np.max(reference_mesh_3d.geometry.x[:, 2])
    facets = mesh.locate_entities_boundary(
        reference_mesh_3d,
        dim=(reference_mesh_3d.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], left)
        | np.isclose(x[0], right)
        | np.isclose(x[1], bottom)
        | np.isclose(x[1], top)
        | np.isclose(x[2], back)
        | np.isclose(x[2], front),
    )
    dofs = fem.locate_dofs_topological(
        V_ref, entity_dim=(reference_mesh_3d.topology.dim - 1), entities=facets
    )
    bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V_ref)
    bcs = [bc]
    lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    u_ref = lp.solve()

    u_phmm = phmm.solve()

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

    u_ref_interpolated = interpolate_nonmatching(u_phmm._V, V_ref, u_ref)
    relative_error = calc_l2_error(u_phmm, u_ref_interpolated) / calc_l2_norm(u_ref_interpolated)

    assert relative_error < atol_3d, (
        f"Relative error in 3D HMM too high. This is a heuristic, to check for code regressions. {relative_error=}"
    )
