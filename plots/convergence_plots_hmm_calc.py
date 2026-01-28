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
from convergence_plots_funcs import (
    EPS,
    HMM_REFINEMENTS,
    A,
    Dtheta,
    f_rhs,
    interpolate_nonmatching,
    l2_relative_error,
    mesh_h_max,
    parallel_print,
)
from dolfinx import fem, mesh
from mpi4py import MPI

from hommx.hmm import PoissonStratifiedHMM

COMM = MPI.COMM_WORLD

parallel_print("Building mesh hierarchy")
# --- Mesh hierarchy ---
msh_base = mesh.create_unit_square(COMM, 4, 4)
msh_base.topology.create_entities(1)
meshes_hmm = [msh_base]
for _ in range(HMM_REFINEMENTS):
    meshes_hmm.append(mesh.refine(meshes_hmm[-1])[0])
    meshes_hmm[-1].topology.create_entities(1)


msh_base_self = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
msh_base_self.topology.create_entities(1)
meshes_hmm_self = [msh_base_self]
for _ in range(HMM_REFINEMENTS):
    meshes_hmm_self.append(mesh.refine(meshes_hmm_self[-1])[0])
    meshes_hmm_self[-1].topology.create_entities(1)


parallel_print("Reading reference solution")
REFERENCE_MESH_FILENAME = "ref_mesh"
msh_ref = adios4dolfinx.read_mesh(REFERENCE_MESH_FILENAME, comm=MPI.COMM_WORLD)
V_ref = fem.functionspace(msh_ref, ("Lagrange", 1))
u_eps_ref = fem.Function(V_ref)
adios4dolfinx.read_function(REFERENCE_MESH_FILENAME, u_eps_ref, name="u_eps_ref")

parallel_print("Read reference mesh and solution")
# --- Convergence study ---

hmm_h: list[float] = []
hmm_err_to_ref: list[float] = []


parallel_print("\nComputing HMM convergence...")
for i, (msh, msh_self) in enumerate(zip(meshes_hmm, meshes_hmm_self)):
    h = mesh_h_max(msh)

    def A_y(y):
        return A(None, y)

    # hmm = PoissonPeriodicHMM(msh, A_y, f_rhs, msh, EPS)
    # hmm = PoissonHMM(msh, A, f_rhs, msh_self, EPS)
    hmm = PoissonStratifiedHMM(msh, A, f_rhs, msh_self, EPS, Dtheta)

    # enforce same BCs as FEM
    # hmm.set_boundary_conditions(zero_dirichlet_bcs(hmm.function_space))
    # hmm._bcs = []
    u_hmm = hmm.solve()

    Vh = hmm.function_space
    u_ref_interp = interpolate_nonmatching(Vh, V_ref, u_eps_ref)
    # u_hmm_interp = interpolate_nonmatching(V_ref, Vh, u_hmm)

    hmm_h.append(h)
    hmm_err_to_ref.append(l2_relative_error(u_ref_interp, u_hmm))

    if COMM.rank == 0:
        parallel_print(
            f"  HMM level {i + 1}/{len(meshes_hmm)}: h={h:.3e}, "
            f"relerr(u_eps_ref)={hmm_err_to_ref[-1]:.3e}, "
        )


# --- Plot ---
if COMM.rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(hmm_h, hmm_err_to_ref, "s-", linewidth=2, label=r"HMM vs $u_{\varepsilon,ref}$")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(rf"Convergence for $\varepsilon={EPS:.0e}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("convergence_hmm_stratified.png", dpi=300, bbox_inches="tight")
    plt.show()


# --- CSV export ---
if COMM.rank == 0:
    with open("convergence_hmm_stratified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "h_hmm",
                "relerr_hmm_vs_u_eps_ref",
            ]
        )
        for k in range(len(hmm_h)):
            row = []
            row += [hmm_h[k], hmm_err_to_ref[k]]
            w.writerow(row)

    print("Wrote convergence_hmm_stratified.csv and convergence_hmm_stratified.png")
