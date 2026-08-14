"""Shared helpers for the integration tests."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

COMM = MPI.COMM_SELF


@pytest.fixture
def interpolate_nonmatching():
    """Returns a helper that interpolates a function onto a non-matching mesh."""

    def _interpolate_nonmatching(
        V_to: fem.FunctionSpace, V_from: fem.FunctionSpace, func: fem.Function
    ) -> fem.Function:
        tdim = V_to.mesh.topology.dim
        cells_V_to = np.arange(V_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        interpolation_data = fem.create_interpolation_data(V_to, V_from, cells_V_to)
        func_V_to = fem.Function(V_to)
        func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
        func_V_to.x.scatter_forward()
        return func_V_to

    return _interpolate_nonmatching


@pytest.fixture
def calc_l2_error():
    """Returns a helper computing the $L^2$ norm of the difference of two functions."""

    def _calc_l2_error(u1, u2):
        return np.sqrt(
            COMM.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(u1 - u2, u1 - u2) * ufl.dx)), op=MPI.SUM
            )
        )

    return _calc_l2_error


@pytest.fixture
def calc_l2_norm():
    """Returns a helper computing the $L^2$ norm of a function."""

    def _calc_l2_norm(u1):
        return np.sqrt(
            COMM.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u1, u1) * ufl.dx)), op=MPI.SUM)
        )

    return _calc_l2_norm
