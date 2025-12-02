# %%
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.hmm import PoissonHMM

eps = 1 / 2**3


def A(x, y):
    return 1.1 + x[0] + ufl.sin(2 * ufl.pi * y[0])
    # return ufl.conditional(ufl.sin(2*ufl.pi*y[0]) > 0, 10, 1)
    # return 5


def f(x):
    return 1


COMM = MPI.COMM_WORLD


# %%
N = 6
nx = N
ny = N
nz = N
msh = mesh.create_box(COMM, np.array([[0, 0, 0], [1, 1, 1]]), [nx, ny, nz])
msh2 = mesh.create_unit_cube(COMM, nx, ny, nz)
msh = mesh.create_unit_cube(COMM, nx, ny, nz)
phmm = PoissonHMM(msh, A, f, msh2, eps, petsc_options_cell_problem={"ksp_atol": 1e-9})

print(msh.topology.index_map(3).size_global)
print(msh.topology.index_map(2).size_global)
print(msh.topology.index_map(0).size_global)


# %%
def A_fem(x):
    return A(x, x / eps)


N_ref = 2**6
nx_ref = N_ref
ny_ref = N_ref
nz_ref = N_ref
msh_ref = mesh.create_box(COMM, np.array([[0, 0, 0], [1, 1, 1]]), [nx_ref, ny_ref, nz_ref])
msh_ref = mesh.create_unit_cube(COMM, nx_ref, ny_ref, nz_ref)
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
back = np.min(msh_ref.geometry.x[:, 2])
front = np.max(msh_ref.geometry.x[:, 2])
facets = mesh.locate_entities_boundary(
    msh_ref,
    dim=(msh_ref.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], left)
    | np.isclose(x[0], right)
    | np.isclose(x[1], bottom)
    | np.isclose(x[1], top)
    | np.isclose(x[2], back)
    | np.isclose(x[2], front),
)
dofs = fem.locate_dofs_topological(V_ref, entity_dim=(msh_ref.topology.dim - 1), entities=facets)
bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V_ref)
bcs = [bc]
lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
u_ref = lp.solve()
# %%

u_phmm = phmm.solve()


# %%
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


# %%
u_ref_interpolated = interpolate_nonmatching(u_phmm._V, V_ref, u_ref)
u_phmm_interpolated = interpolate_nonmatching(V_ref, u_phmm._V, u_phmm)
# %%
print(
    f"relative error of hmm against FEM ref for {eps=}, {N=}, {N_ref=} is {calc_l2_error(u_phmm, u_ref_interpolated) / calc_l2_norm(u_ref_interpolated)}"
)
