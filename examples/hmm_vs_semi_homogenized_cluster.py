# %%
import os
from pathlib import Path

import adios4dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh
from hmm_vs_semi_create_reference_cluster import (
    BOX,
    A,
    Dtheta,
    f,
    parallel_print,
    theta,
)
from mpi4py import MPI

from hommx.helpers import PoissonFEM
from hommx.hmm import PoissonHMM, PoissonStratifiedHMM

# eps = 0.9 * 1 / 2**6
# eps = 0.9 * 1 / 2**7 # cluster
eps_list = [0.9 * 1 / 2**i for i in range(2, 9)]
N_ref = 2 ** (12) + 1
# N_ref = 2 ** (13) + 1 # cluster
N_ref_save = 2**8

# %%
if __name__ == "__main__":
    # %%
    MESH_IN_SLURM_JOB_ID = 4097
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", f"{N_ref}_homogenized")
    mesh_file = Path(f"./reference_mesh_{SLURM_JOB_ID}.bp")
    mesh_in_file = Path(f"./reference_mesh_{MESH_IN_SLURM_JOB_ID}.bp")

    if not mesh_in_file.exists():
        raise FileNotFoundError(
            f"Did not find reference solution and mesh, please create them first. {mesh_file}"
        )

    # %%
    COMM = MPI.COMM_WORLD

    msh_ref = adios4dolfinx.read_mesh(mesh_in_file, comm=COMM)
    adios4dolfinx.write_mesh(
        mesh_file, msh_ref
    )  # always need to rewrite to different path, because different DoF Layouts
    parallel_print(msh_ref.geometry.index_map().size_local)
    V_ref = fem.functionspace(msh_ref, ("Lagrange", 1))
    u_fem_ref_list = []
    for eps in eps_list:
        u_fem_ref = fem.Function(V_ref)
        adios4dolfinx.read_function(mesh_in_file, u_fem_ref, name=f"reference_{N_ref}_{eps}")
        adios4dolfinx.write_function(mesh_file, u_fem_ref, name=f"reference_{N_ref}_{eps}")
        u_fem_ref_list.append(u_fem_ref)
    parallel_print("read reference mesh and fem solutions")

    # %%
    def interpolate_nonmatching(
        V_to: fem.FunctionSpace,
        V_from: fem.FunctionSpace,
        func: fem.Function,
        interpolation_data=None,
        cells_V_to=None,
    ) -> fem.Function:
        tdim = V_to.mesh.topology.dim
        if cells_V_to is None:
            cells_V_to = np.arange(V_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        if interpolation_data is None:
            interpolation_data = fem.create_interpolation_data(V_to, V_from, cells_V_to)
        func_V_to = fem.Function(V_to)
        func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
        func_V_to.x.scatter_forward()
        return func_V_to

    parallel_print("Calculating homogenized solutions")
    Ns = list(range(7, 26))  # + list(range(27, 60, 3))
    Ns = np.array(Ns, dtype=int)  # cluster
    # Ns = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140], dtype=int) # cluster
    # Ns = np.logspace(1, 1.7, 10, dtype=int)
    Vs = []
    phmm_sols = []
    phmmsemi_sols = []
    pfem_sols = []
    phmm_sols_interpolated = []
    phmmsemi_sols_interpolated = []
    pfem_sols_interpolated = []

    for N in Ns:
        nx = N
        msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [BOX, BOX]]), [nx, nx])
        msh2 = mesh.create_unit_square(MPI.COMM_SELF, nx, nx)
        phmm = PoissonHMM(
            msh,
            A,
            f,
            msh2,
            eps,
            petsc_options_cell_problem={"ksp_atol": 1e-9, "ksp_type": "gmres", "pc_type": "ilu"},
        )
        phmmsemi = PoissonStratifiedHMM(
            msh,
            A,
            f,
            msh2,
            eps,
            Dtheta,
            petsc_options_cell_problem={"ksp_atol": 1e-9, "ksp_type": "gmres", "pc_type": "ilu"},
        )

        def A_fem(x):
            return A(x, theta(x) / eps)

        pfem = PoissonFEM(msh, A_fem, f)
        u_phmm = phmm.solve()
        u_phmmsemi = phmmsemi.solve()
        u_fem = pfem.solve()
        Vs.append(phmm._V_macro)
        phmm_sols.append(u_phmm)
        phmmsemi_sols.append(u_phmmsemi)
        pfem_sols.append(u_fem)
        cells_V_to = np.arange(
            V_ref.mesh.topology.index_map(V_ref.mesh.topology.dim).size_local, dtype=np.int32
        )
        interpolation_data = fem.create_interpolation_data(V_ref, phmm._V_macro, cells_V_to)
        phmm_sols_interpolated.append(
            interpolate_nonmatching(
                V_ref,
                phmm._V_macro,
                u_phmm,
                cells_V_to=cells_V_to,
                interpolation_data=interpolation_data,
            )
        )
        phmmsemi_sols_interpolated.append(
            interpolate_nonmatching(
                V_ref,
                phmm._V_macro,
                u_phmmsemi,
                cells_V_to=cells_V_to,
                interpolation_data=interpolation_data,
            )
        )
        pfem_sols_interpolated.append(
            interpolate_nonmatching(
                V_ref,
                phmm._V_macro,
                u_fem,
                cells_V_to=cells_V_to,
                interpolation_data=interpolation_data,
            )
        )
        # parallel_print(phmmsemi_sols_interpolated[-1].x.array.max())
        adios4dolfinx.write_function(mesh_file, phmm_sols_interpolated[-1], name=f"phmm_{N}")
        adios4dolfinx.write_function(
            mesh_file, phmmsemi_sols_interpolated[-1], name=f"phmmsemi_{N}"
        )
        adios4dolfinx.write_function(mesh_file, pfem_sols_interpolated[-1], name=f"pfem_{N}")

    # %% calculate L2 error

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

    for eps, u_fem_ref in zip(eps_list, u_fem_ref_list, strict=True):
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

        # %% calc hmm vs semihmm
        L2_errors_hmm_vs_semi = []
        for u_phmm, u_phmmsemi in zip(phmm_sols, phmmsemi_sols):
            L2_error = calc_l2_error(u_phmm, u_phmmsemi)
            norm = calc_l2_norm(u_phmm)
            L2_errors_hmm_vs_semi.append(L2_error / norm)
        # %%
        # plot errors
        # if COMM.rank == 0:
        #     plt.loglog(1 / Ns, L2_errors_hmm, label="HMM")
        #     plt.loglog(1 / Ns, L2_errors_semihmm, label="SemiHMM")
        #     plt.loglog(1 / Ns, L2_errors_fem, label="FEM")
        #     plt.ylabel(r"$\|u_{fem}-u_{hmm}\|_{L^2}$")
        #     plt.xlabel("h, H")
        #     plt.legend()
        #     plt.savefig(f"errors_hmm_vs_semi_{BOX}_{SLURM_JOB_ID}.pdf")
        #     plt.show()

        # %%
        # plot relative errors
        u_fem_ref_max = COMM.allreduce(u_fem_ref.x.array.max(), op=MPI.MAX)
        if COMM.rank == 0:
            plt.loglog(1 / Ns, L2_errors_hmm / norm_u_fem, label="HMM")
            plt.loglog(1 / Ns, L2_errors_semihmm / norm_u_fem, label="SemiHMM")
            plt.loglog(1 / Ns, L2_errors_fem / norm_u_fem, label="FEM")
            plt.hlines(
                [eps],
                xmin=min(1 / Ns),
                xmax=max(1 / Ns),
                label="eps",
                linestyles=[":"],
                colors=["green"],
            )
            plt.ylabel(r"$\|u_{fem}-u_{hmm}\|_{L^2}/\|u_{fem}\|_{L^2}$")
            plt.xlabel("h, H")
            plt.legend()
            plt.title(f"{eps=}, {u_fem_ref_max=}")
            plt.savefig(f"relative_errors_hmm_vs_semi_{BOX}_{SLURM_JOB_ID}_{eps}.pdf")
            # plt.show()
            plt.close()

    # %%
    # plot hmm vs semi
    if COMM.rank == 0:
        plt.loglog(1 / Ns, L2_errors_hmm_vs_semi)
        plt.savefig(f"L2_errors_hmm_vs_semi_{BOX}_{SLURM_JOB_ID}.pdf")
        plt.show()


# %% save functions to disk
