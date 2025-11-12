import os
import sys
from pathlib import Path

import adios4dolfinx
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

eps = 0.9 * 1 / 2**3
BOX = 5
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "no_slurm")
mesh_file = Path(f"reference_mesh_{SLURM_JOB_ID}.bp")
N_ref = 2 ** (8) + 1


def A(x, y):
    return ufl.conditional(ufl.cos(2 * ufl.pi * y[0]) < 0, 10, 1)


def f(x):
    return 1


theta_factor = 0.2


def theta(x):
    factor = theta_factor * (ufl.cos(ufl.pi / 2 * x[1])) * (ufl.cos(ufl.pi / 2 * x[0]))
    x_0 = x[0] - factor * x[1]
    x_1 = x[1] + factor * x[0]
    return ufl.as_vector([x_0, x_1])


def Dtheta(x):
    arg_0 = ufl.pi / 2 * x[0]
    arg_1 = ufl.pi / 2 * x[1]
    f = theta_factor * ufl.cos(arg_0) * ufl.cos(arg_1)
    df_dx0 = -theta_factor * (ufl.pi / 2) * ufl.sin(arg_0) * ufl.cos(arg_1)
    df_dx1 = -theta_factor * (ufl.pi / 2) * ufl.cos(arg_0) * ufl.sin(arg_1)

    return ufl.as_matrix(
        [[1 - x[1] * df_dx0, f + x[0] * df_dx0], [-f - x[1] * df_dx1, 1 + x[0] * df_dx1]]
    )


def parallel_print(msg: str):
    print(f"Rank {MPI.COMM_WORLD.rank}: {msg}")
    sys.stdout.flush()


# %%
def A_fem(x):
    return A(x, theta(x) / eps)


if __name__ == "__main__":
    msh_ref = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [N_ref, N_ref])
    parallel_print("Writing reference mesh")
    adios4dolfinx.write_mesh(mesh_file, msh_ref)
    V_ref = fem.functionspace(msh_ref, ("Lagrange", 1))
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    x = ufl.SpatialCoordinate(msh_ref)
    lhs = ufl.inner(A_fem(x) * ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(f(x), v) * ufl.dx
    left = np.min(msh_ref.geometry.x[:, 0])
    right = np.max(msh_ref.geometry.x[:, 0])
    bottom = np.min(msh_ref.geometry.x[:, 1])
    top = np.max(msh_ref.geometry.x[:, 1])
    facets = mesh.locate_entities_boundary(
        msh_ref,
        dim=(msh_ref.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], left)
        | np.isclose(x[0], right)
        | np.isclose(x[1], bottom)
        | np.isclose(x[1], top),
    )
    dofs = fem.locate_dofs_topological(V_ref, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V_ref)
    bcs = [bc]

    lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "ilu"})
    parallel_print("Created reference problem")
    u_fem_ref = lp.solve()
    parallel_print("solved reference problem")
    adios4dolfinx.write_function(mesh_file, u_fem_ref, name=f"reference_{eps}_{N_ref}")
    parallel_print("wrote reference solution")
