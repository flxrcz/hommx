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

# eps = 0.9 * 1 / 2**6
# eps = 0.9 * 1 / 2**7 # cluster
eps_list = [0.9 * 1 / 2**i for i in range(2, 9)]
N_ref = 2 ** (12) + 1
# N_ref = 2 ** (13) + 1 # cluster
N_ref_save = 2**8

BOX = 5
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", f"{N_ref}")
mesh_file = Path(f"reference_mesh_{SLURM_JOB_ID}.bp")


def A(x, y):
    # return ufl.conditional(ufl.cos(2 * ufl.pi * y[0]) < 0, 10, 1)
    return 5 + 4.5 * ufl.sin(2 * ufl.pi * y[0])


def f(x):
    return 1


theta_factor = 0.2


def theta(x):
    factor = theta_factor * (ufl.cos(ufl.pi / 2 * x[1])) * (ufl.cos(ufl.pi / 2 * x[0]))
    x_0 = x[0] - factor * x[1]
    x_1 = x[1] + factor * x[0]
    return ufl.as_vector([x_0, x_1])


# actually Dtheta_tranposed
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
if __name__ == "__main__":
    msh_ref = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [N_ref, N_ref])
    parallel_print("Created reference mesh")
    adios4dolfinx.write_mesh(mesh_file, msh_ref)
    V_ref = fem.functionspace(msh_ref, ("Lagrange", 1))
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    x = ufl.SpatialCoordinate(msh_ref)
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

    msh_ref_save = mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [N_ref_save, N_ref_save]
    )
    adios4dolfinx.write_mesh(mesh_file, msh_ref_save)
    parallel_print(f"Saved reference mesh of size {N_ref_save}")
    V_ref_save = fem.functionspace(msh_ref_save, ("Lagrange", 1))
    tdim = V_ref_save.mesh.topology.dim
    cells_V_to = np.arange(V_ref_save.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    interpolation_data = fem.create_interpolation_data(V_ref_save, V_ref, cells_V_to)

    for eps in eps_list:

        def A_fem(x):
            return A(x, theta(x) / eps)

        lhs = ufl.inner(A_fem(x) * ufl.grad(u), ufl.grad(v)) * ufl.dx
        lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
        parallel_print("Created reference problem")
        u_fem_ref = lp.solve()
        parallel_print(u_fem_ref.x.array.max())
        parallel_print(f"solved reference problem size {N_ref} with {eps=}")
        u_fem_ref_save = fem.Function(V_ref_save)
        u_fem_ref_save.interpolate_nonmatching(u_fem_ref, cells_V_to, interpolation_data)
        u_fem_ref_save.x.scatter_forward()
        parallel_print(f"{u_fem_ref_save.x.array.max()=}")
        parallel_print(f"Interpolated to save mesh of size {N_ref_save}")
        adios4dolfinx.write_function(mesh_file, u_fem_ref_save, name=f"reference_{N_ref}_{eps}")
        parallel_print("Saved mesh function.")
