import sys

import numpy as np
import ufl
from dolfinx import cpp, fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

EPS = 2 ** (-7)

FEM_REFINEMENTS = 7  # 5 for testing, 7 for running (means 33mio elements at reference level)
HMM_REFINEMENTS = 4  # number of macro refinement levels to run HMM on (incl. base mesh)
REF_EXTRA_REFINEMENTS = 3  # extra refinements on top of finest FEM mesh for reference u_ε


def parallel_print(msg: str):
    print(f"Rank {MPI.COMM_WORLD.rank}: {msg}")
    sys.stdout.flush()


def mesh_h_max(msh: mesh.Mesh) -> float:
    tdim = msh.topology.dim
    num_cells_local = msh.topology.index_map(tdim).size_local
    if num_cells_local == 0:
        return 0.0
    h_cells = cpp.mesh.h(msh._cpp_object, tdim, np.arange(num_cells_local, dtype=np.int32))
    return float(msh.comm.allreduce(np.max(h_cells), op=MPI.MAX))


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


def l2_error(u1: fem.Function, u2: fem.Function) -> float:
    err_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u1 - u2, u1 - u2) * ufl.dx))
    err_sq = u1.function_space.mesh.comm.allreduce(err_sq_local, op=MPI.SUM)
    return float(np.sqrt(err_sq))


def l2_norm(u: fem.Function) -> float:
    n_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx))
    n_sq = u.function_space.mesh.comm.allreduce(n_sq_local, op=MPI.SUM)
    return float(np.sqrt(n_sq))


def l2_relative_error(u_ref: fem.Function, u_approx: fem.Function) -> float:
    denom = l2_norm(u_ref)
    if denom == 0.0:
        return float("nan")
    return l2_error(u_ref, u_approx) / denom


def zero_dirichlet_bcs(V: fem.FunctionSpace) -> list[fem.DirichletBC]:
    msh = V.mesh
    tdim = msh.topology.dim
    left = np.min(msh.geometry.x[:, 0])
    right = np.max(msh.geometry.x[:, 0])
    bottom = np.min(msh.geometry.x[:, 1])
    top = np.max(msh.geometry.x[:, 1])
    facets = mesh.locate_entities_boundary(
        msh,
        dim=(tdim - 1),
        marker=lambda x: np.isclose(x[0], left)
        | np.isclose(x[0], right)
        | np.isclose(x[1], bottom)
        | np.isclose(x[1], top),
    )
    dofs = fem.locate_dofs_topological(V, entity_dim=(tdim - 1), entities=facets)
    return [fem.dirichletbc(PETSc.ScalarType(0), dofs, V)]


# --- Analytical test case (from integration test) ---
def A(x, y):
    # return 5 + 4.8*ufl.sin(2 * ufl.pi * y[0])
    return 1 / (2 + ufl.cos(2 * ufl.pi * y[0]))


def f_rhs(x):
    return ufl.pi**2 * (1 / 2 + 1 / ufl.sqrt(3)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


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


def A_eps(x, eps: float):
    return A(x, theta(x) / eps)


# def A_theta_eps(x, eps: float):
#     return A(x, theta(x)/eps)
