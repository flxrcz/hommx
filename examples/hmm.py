# %%
import ufl
from dolfinx import mesh
from mpi4py import MPI

from hommx.hmm import PoissonHMM

eps = 1/2**5
def A(x):
    def A(y):
        return 1 + 0.5*ufl.sin(2*ufl.pi*y[0]) + 0.5*ufl.sin(2*ufl.pi*y[1])
    return A
f=1

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
