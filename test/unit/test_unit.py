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


@pytest.fixture
def unit_cube_functionspace():
    nx = ny = nz = 10
    msh = mesh.create_unit_cube(MPI.COMM_SELF, nx, ny, nz)
    V = fem.functionspace(msh, ("Lagrange", 1))
    return V


def test_periodic_boundary_conditions_unit_square(unit_square_functionspace):
    V = unit_square_functionspace
    msh = V.mesh
    mpc = create_periodic_boundary_conditions(V)

    verts = mesh.locate_entities_boundary(msh, 0, lambda x: np.ones((x.shape[1],)))
    boundary_dofs = fem.locate_dofs_topological(V, entity_dim=0, entities=verts)
    points = V.tabulate_dof_coordinates()

    for dof in range(len(points)):
        # only boundary dofs should be constrained
        if dof not in boundary_dofs:
            assert dof not in mpc.slaves, "Non boundary DoF should not be constrained"

        # special DoF (we force u(1,1) = u(0,0)), since mpc does not allow nested bcs
        if np.allclose(points[dof], np.array([1, 1, 0])):
            assert dof in mpc.slaves
            master = mpc.masters.links(dof)
            assert np.allclose(points[master], np.array([0, 0, 0])), (
                "DoF at (1, 1) should be slave of DoF at (0, 0)"
            )
            continue

        if dof in mpc.slaves:
            master = mpc.masters.links(dof)
            assert master in boundary_dofs, "Master DoF should also be a boundary DoF"
            diff = np.abs(points[master] - points[dof])
            assert np.allclose(diff, np.array([1, 0, 0])) | np.allclose(
                diff, np.array([0, 1, 0])
            ), f"Master slave relation violated, master: {points[master]}, slave: {points[dof]} "


def test_periodic_boundary_conditions_unit_cube(unit_cube_functionspace):
    V = unit_cube_functionspace
    msh = V.mesh
    mpc = create_periodic_boundary_conditions(V)

    verts = mesh.locate_entities_boundary(msh, 0, lambda x: np.ones((x.shape[1],)))
    boundary_dofs = fem.locate_dofs_topological(V, entity_dim=0, entities=verts)
    points = V.tabulate_dof_coordinates()

    for dof in range(len(points)):
        # only boundary dofs should be constrained
        if dof not in boundary_dofs:
            assert dof not in mpc.slaves, "Non boundary DoF should not be constrained"

        # special DoF (we force u(1,1,1) = u(0,0,0)), since mpc does not allow nested bcs
        if np.allclose(points[dof], np.array([1, 1, 1])):
            assert dof in mpc.slaves
            master = int(
                mpc.masters.links(dof)[0]
            )  # cast to int to make sure points[master] has correct shape
            assert np.allclose(points[master], np.array([0, 0, 0])), (
                "DoF at (1, 1) should be slave of DoF at (0, 0)"
            )
            continue

        # DoFs that are constrained twice
        handled = False
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if np.allclose(points[dof][[i, j]], np.array([1, 1])):
                assert dof in mpc.slaves
                master = int(mpc.masters.links(dof)[0])
                assert np.allclose(points[master][[i, j]], np.array([0, 0])), (
                    f"DoF at {points[dof]} should be slave of another DoF, but is slave of {points[master]}"
                )
                handled = True
        if handled:
            continue

        if dof in mpc.slaves:
            master = int(mpc.masters.links(dof)[0])
            assert master in boundary_dofs, "Master DoF should also be a boundary DoF"
            diff = np.abs(points[master] - points[dof])
            assert (
                np.allclose(diff, np.array([1, 0, 0]))
                | np.allclose(diff, np.array([0, 1, 0]))
                | np.allclose(diff, np.array([0, 0, 1]))
            ), f"Master slave relation violated, master: {points[master]}, slave: {points[dof]} "
