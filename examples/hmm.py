# %%
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from hommx.hmm import PoissonHMM

eps = 1 / 2**5


def A(x):
    def A(y):
        return 1 / (2 + ufl.cos(2 * ufl.pi * y[0]))

    return A


def f(x):
    return ufl.pi**2 * (1 / 2 + 1 / ufl.sqrt(3)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def solution(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


# %%
N = 15
nx = N
ny = N
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
msh2 = mesh.create_unit_square(MPI.COMM_SELF, nx, nx)
phmm = PoissonHMM(msh, A, f, msh2, eps, petsc_options_cell_problem={"ksp_atol": 1e-10})
print(msh.topology.index_map(2).size_global)
print(msh.topology.index_map(0).size_global)
# %%

phmm.solve()

# %%
phmm.plot_solution()

# %%
u = fem.Function(phmm._V_macro)
u.interpolate(solution)

# %%
