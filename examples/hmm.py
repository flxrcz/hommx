# %%
import ufl
from dolfinx import mesh
from mpi4py import MPI

from hommx.hmm import PoissonHMM

eps = 1 / 2**5


def A(x, y):
    return 0.33 + 0.15 * (ufl.sin(2 * ufl.pi * x[0]) + ufl.sin(2 * ufl.pi * y[0]))


def f(x):
    return (
        3.25696945235949
        * ufl.sqrt((0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1) ** 2 - 0.206611570247934)
        * ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[1])
        + ufl.pi**2
        * (0.15 * ufl.sin(2 * ufl.pi * x[0]) + 0.33)
        * ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[1])
        - 2.96088132032681
        * (0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1)
        * ufl.sin(ufl.pi * x[1])
        * ufl.cos(ufl.pi * x[0])
        * ufl.cos(2 * ufl.pi * x[0])
        / ufl.sqrt((0.454545454545455 * ufl.sin(2 * ufl.pi * x[0]) + 1) ** 2 - 0.206611570247934)
    )


def solution(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


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
# u = fem.Function(phmm._V_macro)
# u.interpolate(solution)

# %%
