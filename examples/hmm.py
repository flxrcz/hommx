# %%
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.hmm import PoissonHMM

eps = 1 / 2**5


def A(x, y):
    return 1.1 + x[0] + ufl.sin(2 * ufl.pi * y[0])
    # return ufl.conditional(x[0] > 0.5, 10, 1)
    # return ufl.conditional(ufl.sin(2*ufl.pi*y[0]) > 0, 10, 1)
    # return 5


def f(x):
    return fem.Constant(x.ufl_domain(), PETSc.ScalarType(0))


COMM = MPI.COMM_WORLD


# %%
N = 15
nx = N
ny = N
msh = mesh.create_rectangle(COMM, np.array([[0, 0], [5, 5]]), [nx, nx])
msh2 = mesh.create_unit_square(COMM, nx, nx)
phmm = PoissonHMM(msh, A, f, msh2, eps, petsc_options_cell_problem={"ksp_atol": 1e-9})
left = np.min(msh.geometry.x[:, 0])
right = np.max(msh.geometry.x[:, 0])
bottom = np.min(msh.geometry.x[:, 1])
top = np.max(msh.geometry.x[:, 1])
facets_left = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], left),
)
facets_right = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], right),
)
dofs_left = fem.locate_dofs_topological(
    phmm.function_space, entity_dim=(msh.topology.dim - 1), entities=facets_left
)
bc_left = fem.dirichletbc(value=PETSc.ScalarType(1), dofs=dofs_left, V=phmm.function_space)

dofs_right = fem.locate_dofs_topological(
    phmm.function_space, entity_dim=(msh.topology.dim - 1), entities=facets_right
)
bc_right = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs_right, V=phmm.function_space)

phmm.set_boundary_condtions([bc_left, bc_right])


print(msh.topology.index_map(2).size_global)
print(msh.topology.index_map(0).size_global)


# %%
def A_fem(x):
    return A(x, x / eps)


N = 2**7
nx = N
ny = N
# msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [5, 5]]), [nx, nx])
V_ref = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V_ref)
v = ufl.TestFunction(V_ref)
x = ufl.SpatialCoordinate(msh)
lhs = ufl.inner(A_fem(x) * ufl.grad(u), ufl.grad(v)) * ufl.dx
rhs = ufl.inner(f(x), v) * ufl.dx
left = np.min(msh.geometry.x[:, 0])
right = np.max(msh.geometry.x[:, 0])
bottom = np.min(msh.geometry.x[:, 1])
top = np.max(msh.geometry.x[:, 1])
facets_left = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], left),
)
facets_right = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], right),
)
dofs_left = fem.locate_dofs_topological(
    V_ref, entity_dim=(msh.topology.dim - 1), entities=facets_left
)
bc_left = fem.dirichletbc(value=PETSc.ScalarType(1), dofs=dofs_left, V=V_ref)

dofs_right = fem.locate_dofs_topological(
    V_ref, entity_dim=(msh.topology.dim - 1), entities=facets_right
)
bc_right = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs_right, V=V_ref)
bcs = [bc_left, bc_right]
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
# %%
print(
    f"relative error of hmm against FEM ref for {eps=}, {N=} is {calc_l2_error(u_phmm, u_ref_interpolated) / calc_l2_norm(u_ref_interpolated)}"
)


# %%


cells, types, x = plot.vtk_mesh(V_ref)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u_ref.x.array
grid.set_active_scalars("u")
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)

plotter.show()

# %%
