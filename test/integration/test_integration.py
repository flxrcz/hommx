import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

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
    assert np.isclose(L2_error, 0, atol=atol), f"L^2 error to big {L2_error=}"


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
    assert np.isclose(L2_error, 0, atol=atol), f"L^2 error to big {L2_error=}"
