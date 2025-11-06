# %%
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.helpers import PoissonFEM
from hommx.hmm import PoissonHMM, PoissonSemiHMM

eps = 1 / 2**5
BOX = 5


# def A(x, y):
#     return 0.33 + 0.15 * (ufl.sin(2 * ufl.pi * x[0]) + ufl.sin(2 * ufl.pi * y[0]))


def A(x, y):
    # return 1
    return ufl.conditional(ufl.sin(2 * ufl.pi * y[0]) > 0, 10, 1)


def f(x):
    return 1


def solution(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


theta_factor = 0.2


def theta(x):
    factor = theta_factor * (ufl.cos(ufl.pi / 2 * x[1])) * (ufl.cos(ufl.pi / 2 * x[0]))
    x_0 = x[0] - factor * x[1]
    x_1 = x[1] + factor * x[0]
    return ufl.as_vector([x_0, x_1])


# def Dtheta(x):
#     arg_0 = ufl.pi/2 * x[0]
#     arg_1 = ufl.pi/2 * x[1]
#     f = theta_factor * ufl.cos(arg_0) * ufl.cos(arg_1)
#     df_dx0 = -theta_factor * (ufl.pi/2) * ufl.sin(arg_0) * ufl.cos(arg_1)
#     df_dx1 = -theta_factor * (ufl.pi/2) * ufl.cos(arg_0) * ufl.sin(arg_1)

#     return ufl.as_matrix([
#         [1 - x[1] * df_dx0, -f - x[1] * df_dx1],
#         [f + x[0] * df_dx0, 1 + x[0] * df_dx1]
#     ])


def Dtheta(x):
    arg_0 = ufl.pi / 2 * x[0]
    arg_1 = ufl.pi / 2 * x[1]
    f = theta_factor * ufl.cos(arg_0) * ufl.cos(arg_1)
    df_dx0 = -theta_factor * (ufl.pi / 2) * ufl.sin(arg_0) * ufl.cos(arg_1)
    df_dx1 = -theta_factor * (ufl.pi / 2) * ufl.cos(arg_0) * ufl.sin(arg_1)

    return ufl.as_matrix(
        [[1 - x[1] * df_dx0, f + x[0] * df_dx0], [-f - x[1] * df_dx1, 1 + x[0] * df_dx1]]
    )


# def theta(x):
#     x_0 = x[1]
#     x_1 = x[0]
#     return ufl.as_vector([x_0, x_1])

# def Dtheta(x):
#     return ufl.as_matrix([[1,0],[0,1]])


# %%
# calculate reference solution
print("Calculating reference solution")


def A_fem(x):
    return A(x, theta(x) / eps)
    # return A(x, x/eps)


N_ref = 2 ** (11)
# msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
msh_ref = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [N_ref, N_ref])
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
lp = LinearProblem(lhs, rhs, bcs)
u_fem_ref = lp.solve()
# def Dtheta(x):
#     return ufl.as_matrix([[1,0],[0,1]])


# %%
print("Calculating homogenized solutions")
Ns = np.array([3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50], dtype=int)
# Ns = np.logspace(1, 1.7, 10, dtype=int)
Vs = []
phmm_sols = []
phmmsemi_sols = []
pfem_sols = []
for N in Ns:
    nx = N
    ny = N
    # msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
    msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [nx, nx])
    msh2 = mesh.create_unit_square(MPI.COMM_SELF, nx, nx)
    phmm = PoissonHMM(msh, A, f, msh2, eps, petsc_options_cell_problem={"ksp_atol": 1e-9})
    phmmsemi = PoissonSemiHMM(
        msh, A, f, msh2, eps, Dtheta, petsc_options_cell_problem={"ksp_atol": 1e-9}
    )
    pfem = PoissonFEM(msh, A_fem, f)
    u_phmm = phmm.solve()
    u_phmmsemi = phmmsemi.solve()
    u_fem = pfem.solve()
    Vs.append(phmm._V_macro)
    phmm_sols.append(u_phmm)
    phmmsemi_sols.append(u_phmmsemi)
    pfem_sols.append(u_fem)


# %%
# interpolate hmm onto fine grid
print("Interpolating onto fine grid")


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


phmm_sols_interpolated = []
phmmsemi_sols_interpolated = []
pfem_sols_interpolated = []

for V, phmm_sol, phmmsemi_sol, pfem_sol in zip(Vs, phmm_sols, phmmsemi_sols, pfem_sols):
    phmm_sols_interpolated.append(interpolate_nonmatching(V_ref, V, phmm_sol))
    phmmsemi_sols_interpolated.append(interpolate_nonmatching(V_ref, V, phmmsemi_sol))
    pfem_sols_interpolated.append(interpolate_nonmatching(V_ref, V, pfem_sol))

# %%
# calculate L2 error
COMM = MPI.COMM_WORLD


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


norm_u_fem = calc_l2_norm(u_fem_ref)

L2_errors_hmm = []
L2_errors_semihmm = []
L2_errors_fem = []

for u_phmm, u_phmmsemi, u_fem in zip(
    phmm_sols_interpolated, phmmsemi_sols_interpolated, pfem_sols_interpolated
):
    L2_errors_hmm.append(calc_l2_error(u_phmm, u_fem_ref))
    L2_errors_semihmm.append(calc_l2_error(u_phmmsemi, u_fem_ref))
    L2_errors_fem.append(calc_l2_error(u_fem, u_fem_ref))

L2_errors_hmm = np.asarray(L2_errors_hmm)
L2_errors_semihmm = np.asarray(L2_errors_semihmm)
L2_errors_fem = np.asarray(L2_errors_fem)

# %%
# calc hmm vs semihmm
L2_errors_hmm_vs_semi = []
for u_phmm, u_phmmsemi in zip(phmm_sols, phmmsemi_sols):
    L2_error = calc_l2_error(u_phmm, u_phmmsemi)
    norm = calc_l2_norm(u_phmm)
    L2_errors_hmm_vs_semi.append(L2_error / norm)
# %%
# plot errors
if COMM.rank == 0:
    plt.loglog(1 / Ns, L2_errors_hmm, label="HMM")
    plt.loglog(1 / Ns, L2_errors_semihmm, label="SemiHMM")
    plt.loglog(1 / Ns, L2_errors_fem, label="FEM")
    plt.ylabel(r"$\|u_{fem}-u_{hmm}\|_{L^2}$")
    plt.xlabel("h, H")
    plt.legend()
    plt.savefig(f"errors_hmm_vs_semi_{BOX}.pdf")
    plt.show()

# %%
# plot relative errors
if COMM.rank == 0:
    plt.loglog(1 / Ns, L2_errors_hmm / norm_u_fem, label="HMM")
    plt.loglog(1 / Ns, L2_errors_semihmm / norm_u_fem, label="SemiHMM")
    plt.loglog(1 / Ns, L2_errors_fem / norm_u_fem, label="FEM")
    plt.hlines(
        [eps], xmin=min(1 / Ns), xmax=max(1 / Ns), label="eps", linestyles=[":"], colors=["green"]
    )
    plt.ylabel(r"$\|u_{fem}-u_{hmm}\|_{L^2}/\|u_{fem}\|_{L^2}$")
    plt.xlabel("h, H")
    plt.legend()
    plt.savefig(f"relative_errors_hmm_vs_semi_{BOX}.pdf")
    plt.show()


# %%
# plot hmm vs semi
if COMM.rank == 0:
    plt.loglog(1 / Ns, L2_errors_hmm_vs_semi)
    plt.show()
# %%
# from dolfinx import plot
# import pyvista as pv

# cells, types, x = plot.vtk_mesh(V_ref)
# grid = pv.UnstructuredGrid(cells, types, x)
# grid.point_data["u"] = u_fem_ref.x.array
# grid.set_active_scalars("u")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(grid, show_edges=True)
# warped = grid.warp_by_scalar()
# plotter.add_mesh(warped)

# plotter.show()
# %%
