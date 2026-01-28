"""Convergence plot for the analytical Poisson homogenization example.

We fix one microscopic scale ε and compare, as h → 0:


- standard FEM with oscillatory coefficient A(x/ε)
- HMM solution on macro mesh with periodic micro cell problems

against two targets:
1) a very fine oscillatory FEM reference u_ε,ref
2) the analytical homogenized solution u_0(x) = sin(pi x) sin(pi y)

from hommx.hmm import PoissonHMM
"""

# %%
import csv

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import cpp, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.hmm import PoissonPeriodicHMM

COMM = MPI.COMM_WORLD

# --- Parameters ---
# User request said eps=2**6; in homogenization ε is typically small, so default is 2**(-6).
EPS = 2 ** (-6)

FEM_REFINEMENTS = 5  # number of refinement steps for FEM convergence
HMM_LEVELS = 5  # number of macro refinement levels to run HMM on (incl. base mesh)
REF_EXTRA_REFINEMENTS = 3  # extra refinements on top of finest FEM mesh for reference u_ε


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
def A_micro(y):
    return 1 / (2 + ufl.cos(2 * ufl.pi * y[0]))


def A_hmm(x_macro, y_micro):
    # independent of the slow variable
    return A_micro(y_micro)


def f_rhs(x):
    return ufl.pi**2 * (1 / 2 + 1 / ufl.sqrt(3)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def u0_exact_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def A_eps(x, eps: float):
    # oscillatory coefficient A(x/eps)
    return A_micro(x / eps)


# --- Mesh hierarchy ---
msh_base = mesh.create_unit_square(COMM, 4, 4)
msh_base.topology.create_entities(1)
meshes_fem = [msh_base]
for _ in range(FEM_REFINEMENTS):
    meshes_fem.append(mesh.refine(meshes_fem[-1])[0])
    meshes_fem[-1].topology.create_entities(1)

HMM_LEVELS = min(HMM_LEVELS, len(meshes_fem))
meshes_hmm = meshes_fem[:HMM_LEVELS]

# reference mesh = finest FEM mesh + a few extra refinements
msh_ref = meshes_fem[-1]
for _ in range(REF_EXTRA_REFINEMENTS):
    msh_ref = mesh.refine(msh_ref)[0]
    msh_ref.topology.create_entities(1)

if COMM.rank == 0:
    print(f"eps={EPS:.3e}")
    print(f"reference mesh h={mesh_h_max(msh_ref):.3e}")
    print(
        f"Number of elements on reference mesh: {msh_ref.topology.index_map(tdim := msh_ref.topology.dim).size_local}"
    )


# --- Reference oscillatory FEM solution u_eps_ref on msh_ref ---
print("Computing fine oscillatory FEM reference u_eps_ref...")
V_ref = fem.functionspace(msh_ref, ("Lagrange", 1))
u = ufl.TrialFunction(V_ref)
v = ufl.TestFunction(V_ref)
x_ref = ufl.SpatialCoordinate(msh_ref)
a_ref = ufl.inner(A_eps(x_ref, EPS) * ufl.grad(u), ufl.grad(v)) * ufl.dx
L_ref = ufl.inner(f_rhs(x_ref), v) * ufl.dx
problem_ref = LinearProblem(
    a_ref,
    L_ref,
    bcs=zero_dirichlet_bcs(V_ref),
    petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
)
u_eps_ref = problem_ref.solve()

# --- Convergence study ---
fem_h: list[float] = []
fem_err_to_ref: list[float] = []
fem_err_to_u0: list[float] = []

hmm_h: list[float] = []
hmm_err_to_ref: list[float] = []
hmm_err_to_u0: list[float] = []

print("\nComputing FEM convergence...")
for i, msh in enumerate(meshes_fem):
    h = mesh_h_max(msh)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    a = ufl.inner(A_eps(x, EPS) * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_rhs(x), v) * ufl.dx
    problem = LinearProblem(
        a, L, bcs=zero_dirichlet_bcs(V), petsc_options={"ksp_type": "cg", "pc_type": "gamg"}
    )
    u_fem = problem.solve()

    u_ref_interp = interpolate_nonmatching(V, V_ref, u_eps_ref)
    u0 = fem.Function(V)
    u0.interpolate(fem.Expression(u0_exact_expr(x), V.element.interpolation_points()))
    u0.x.scatter_forward()

    fem_h.append(h)
    fem_err_to_ref.append(l2_relative_error(u_ref_interp, u_fem))
    fem_err_to_u0.append(l2_relative_error(u0, u_fem))

    if COMM.rank == 0:
        print(
            f"  FEM level {i + 1}/{len(meshes_fem)}: h={h:.3e}, "
            f"relerr(u_eps_ref)={fem_err_to_ref[-1]:.3e}, "
            f"relerr(u0)={fem_err_to_u0[-1]:.3e}"
        )


print("\nComputing HMM convergence...")
for i, msh in enumerate(meshes_hmm):
    h = mesh_h_max(msh)
    hmm = PoissonPeriodicHMM(msh, A_micro, f_rhs, msh, EPS)
    # enforce same BCs as FEM
    hmm.set_boundary_conditions(zero_dirichlet_bcs(hmm.function_space))
    u_hmm = hmm.solve()

    Vh = hmm.function_space
    u_ref_interp = interpolate_nonmatching(Vh, V_ref, u_eps_ref)
    x = ufl.SpatialCoordinate(msh)
    u0 = fem.Function(Vh)
    u0.interpolate(fem.Expression(u0_exact_expr(x), Vh.element.interpolation_points()))
    u0.x.scatter_forward()

    hmm_h.append(h)
    hmm_err_to_ref.append(l2_relative_error(u_ref_interp, u_hmm))
    hmm_err_to_u0.append(l2_relative_error(u0, u_hmm))

    if COMM.rank == 0:
        print(
            f"  HMM level {i + 1}/{len(meshes_hmm)}: h={h:.3e}, "
            f"relerr(u_eps_ref)={hmm_err_to_ref[-1]:.3e}, "
            f"relerr(u0)={hmm_err_to_u0[-1]:.3e}"
        )


# --- Plot ---
if COMM.rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(fem_h, fem_err_to_ref, "o-", linewidth=2, label=r"FEM vs $u_{\varepsilon,ref}$")
    ax.loglog(fem_h, fem_err_to_u0, "o--", linewidth=2, label=r"FEM vs $u_0$")
    ax.loglog(hmm_h, hmm_err_to_ref, "s-", linewidth=2, label=r"HMM vs $u_{\varepsilon,ref}$")
    ax.loglog(hmm_h, hmm_err_to_u0, "s--", linewidth=2, label=r"HMM vs $u_0$")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(rf"Convergence for $\varepsilon={EPS:.0e}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("convergence_analytic_fem_vs_hmm.png", dpi=300, bbox_inches="tight")
    plt.show()


# --- CSV export ---
if COMM.rank == 0:
    with open("convergence_analytic_fem_vs_hmm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "h_fem",
                "relerr_fem_vs_u_eps_ref",
                "relerr_fem_vs_u0",
                "h_hmm",
                "relerr_hmm_vs_u_eps_ref",
                "relerr_hmm_vs_u0",
            ]
        )
        n = max(len(fem_h), len(hmm_h))
        for k in range(n):
            row = []
            if k < len(fem_h):
                row += [fem_h[k], fem_err_to_ref[k], fem_err_to_u0[k]]
            else:
                row += ["", "", ""]
            if k < len(hmm_h):
                row += [hmm_h[k], hmm_err_to_ref[k], hmm_err_to_u0[k]]
            else:
                row += ["", "", ""]
            w.writerow(row)

    print("Wrote convergence_analytic_fem_vs_hmm.csv and convergence_analytic_fem_vs_hmm.png")
