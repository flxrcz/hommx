
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from hommx.helpers import create_periodic_boundary_conditions


@pytest.fixture
def unit_square_functionspace():
    nx = ny = 10
    msh = mesh.create_unit_square(MPI.COMM_SELF, nx, ny)
    V = fem.functionspace(msh, ("Lagrange", 1))
    return V

def test_periodic_boundary_conditions_unit_square(unit_square_functionspace):
    V = unit_square_functionspace
    msh = V.mesh
    mpc = create_periodic_boundary_conditions(msh, V)

    boundary_dofs = mesh.locate_entities_boundary(msh, 0, lambda x: np.ones((x.shape[1], )))
    points = V.tabulate_dof_coordinates()


    for dof in range(len(points)):
        # only boundary dofs should be constrained
        if dof not in boundary_dofs:
            assert dof not in mpc.slaves, "Non boundary DoF should not be constrained"

        # special DoF (we force u(1,1) = u(0,0)), since mpc does not allow nested bcs
        if np.allclose(points[dof], np.array([1, 1, 0])):
            assert dof in mpc.slaves
            master = mpc.masters.links(dof)
            assert np.allclose(points[master], np.array([0, 0, 0])), "DoF at (1, 1) should be slave of DoF at (0, 0)"
            continue

        if dof in mpc.slaves:
            master = mpc.masters.links(dof)
            assert master in boundary_dofs, "Master DoF should also be a boundary DoF"
            diff = np.abs(points[master] - points[dof])
            assert np.allclose(diff, np.array([1, 0, 0])) | np.allclose(diff, np.array([0, 1, 0])), f"Master slave relation violated, master: {points[master]}, slave: {points[dof]} "
