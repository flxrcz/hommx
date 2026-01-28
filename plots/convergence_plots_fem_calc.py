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

import adios4dolfinx
import matplotlib.pyplot as plt
import ufl
from convergence_plots_funcs import (
    EPS,
    FEM_REFINEMENTS,
    REF_EXTRA_REFINEMENTS,
    A_eps,
    f_rhs,
    interpolate_nonmatching,
    l2_relative_error,
    mesh_h_max,
    parallel_print,
    zero_dirichlet_bcs,
)
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

COMM = MPI.COMM_WORLD

# --- Mesh hierarchy ---
msh_base = mesh.create_unit_square(COMM, 4, 4)
msh_base.topology.create_entities(1)
meshes_fem = [msh_base]
for _ in range(FEM_REFINEMENTS):
    meshes_fem.append(mesh.refine(meshes_fem[-1])[0])
    meshes_fem[-1].topology.create_entities(1)


parallel_print("Created mesh hierarchy")
# reference mesh = finest FEM mesh + a few extra refinements
msh_ref = meshes_fem[-1]
for _ in range(REF_EXTRA_REFINEMENTS):
    msh_ref = mesh.refine(msh_ref)[0]
    msh_ref.topology.create_entities(1)

if COMM.rank == 0:
    parallel_print(f"eps={EPS:.3e}")
    parallel_print(f"reference mesh h={mesh_h_max(msh_ref):.3e}")
    parallel_print(
        f"Number of elements on reference mesh: {msh_ref.topology.index_map(tdim := msh_ref.topology.dim).size_local}"
    )
    parallel_print(f"Number of DoFs on reference mesh: {msh_ref.topology.index_map(0).size_global}")


parallel_print("Created reference mesh")
# --- Reference oscillatory FEM solution u_eps_ref on msh_ref ---
parallel_print("Computing fine oscillatory FEM reference u_eps_ref...")
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
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
)
u_eps_ref = problem_ref.solve()

# save reference solution and mesh
adios4dolfinx.write_mesh("ref_mesh", msh_ref)
adios4dolfinx.write_function("ref_mesh", u=u_eps_ref, name="u_eps_ref")

# --- Convergence study ---
fem_h: list[float] = []
fem_err_to_ref: list[float] = []

parallel_print("\nComputing FEM convergence...")
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
    # u_fem_interp = interpolate_nonmatching(V_ref, V, u_fem)

    fem_h.append(h)
    fem_err_to_ref.append(l2_relative_error(u_ref_interp, u_fem))

    parallel_print(
        f"  FEM level {i + 1}/{len(meshes_fem)}: h={h:.3e}, "
        f"relerr(u_eps_ref)={fem_err_to_ref[-1]:.3e}, "
    )

# --- Plot ---
if COMM.rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(fem_h, fem_err_to_ref, "o-", linewidth=2, label=r"FEM vs $u_{\varepsilon,ref}$")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(rf"Convergence for $\varepsilon={EPS:.0e}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("convergence_fem_stratified.png", dpi=300, bbox_inches="tight")
    plt.show()


# --- CSV export ---
if COMM.rank == 0:
    with open("convergence_fem_stratified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "h_fem",
                "relerr_fem_vs_u_eps_ref",
            ]
        )
        for k in range(len(fem_h)):
            row = []
            row += [fem_h[k], fem_err_to_ref[k]]
            w.writerow(row)

    print("Wrote convergence_fem_stratified.csv and convergence_fem.png")
