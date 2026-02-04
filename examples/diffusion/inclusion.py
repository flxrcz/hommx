# %%
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.hmm import PoissonStratifiedHMM


def func_to_png(
    filename, V, v, save=False, cmap="viridis", vmin=None, vmax=2.3550577798756827, title=None
):
    cells, types, x = plot.vtk_mesh(V)
    values = v.x.array
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    print(vmin)
    print(vmax)
    cells = cells.reshape((-1, 4))[:, 1:]
    triang = tri.Triangulation(x[:, 0], x[:, 1], cells)

    fig, ax = plt.subplots(figsize=(10, 10))
    levels = np.linspace(vmin, vmax, 101)
    mappable = ax.tricontourf(triang, values, levels=levels, cmap=cmap)

    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    fig.colorbar(mappable, ax=ax)
    plt.tight_layout()
    if save:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


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


def darcy_flow_bcs(V: fem.FunctionSpace) -> list[fem.DirichletBC]:
    msh = V.mesh
    tdim = msh.topology.dim
    left = np.min(msh.geometry.x[:, 0])
    right = np.max(msh.geometry.x[:, 0])

    # Left boundary: value = 1
    facets_left = mesh.locate_entities_boundary(
        msh,
        dim=(tdim - 1),
        marker=lambda x: np.isclose(x[0], left),
    )
    dofs_left = fem.locate_dofs_topological(V, entity_dim=(tdim - 1), entities=facets_left)
    bc_left = fem.dirichletbc(PETSc.ScalarType(1), dofs_left, V)

    # Right boundary: value = 0
    facets_right = mesh.locate_entities_boundary(
        msh,
        dim=(tdim - 1),
        marker=lambda x: np.isclose(x[0], right),
    )
    dofs_right = fem.locate_dofs_topological(V, entity_dim=(tdim - 1), entities=facets_right)
    bc_right = fem.dirichletbc(PETSc.ScalarType(0), dofs_right, V)

    return [bc_left, bc_right]


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


def f(x):
    return fem.Constant(x.ufl_domain(), PETSc.ScalarType(1))


def ufl_circle_indicator(x, y):
    # 1-periodic "wrapped distance" around 1/2 in each coordinate.
    # Using 2*pi here (instead of pi) makes the pattern repeat every 1.
    dx = ufl.acos(ufl.cos(2 * ufl.pi * (x - 1 / 2)))
    dy = ufl.acos(ufl.cos(2 * ufl.pi * (y - 1 / 2)))
    # Keep the same physical radius r=1/4: dx≈2*pi*|x-1/2| near the center.
    r = 0.25
    return (dx**2 + dy**2) < ((2 * ufl.pi) ** 2 * (r) ** 2)


def A(x, y):
    return ufl.conditional(ufl_circle_indicator(y[0], y[1]), 0.001, 0.1)


def theta(x):
    factor = 0.5 * ufl.sin(2 * ufl.pi * x[1])
    x_0 = x[0] + factor
    x_1 = x[1]
    return ufl.as_vector([x_0, x_1])


def Dtheta(x):
    df_1_x_1 = 1
    df_1_x_2 = 0.5 * 2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1])
    df_2_x_1 = 0
    df_2_x_2 = 1
    D = ufl.as_matrix([[df_1_x_1, df_1_x_2], [df_2_x_1, df_2_x_2]])
    return ufl.transpose(D)


eps = 2 ** (-5)


def A_fem(x):
    return A(x, theta(x) / eps)


COMM = MPI.COMM_WORLD


# %%
N = 2 ** (10)
nx = N
ny = N

msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
V_ref = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V_ref)
v = ufl.TestFunction(V_ref)
x = ufl.SpatialCoordinate(msh)
lhs = ufl.inner(A_fem(x) * ufl.grad(u), ufl.grad(v)) * ufl.dx
rhs = ufl.inner(f(x), v) * ufl.dx
bcs = darcy_flow_bcs(V_ref)
lp = LinearProblem(lhs, rhs, bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
u_ref = lp.solve()


func_to_png("diffusion_wavy_inclusion_reference", V_ref, u_ref, save=True, title="$u_\\text{ref}$")


# change coefficient for coefficient plot
eps_plot = 2 ** (-4)


def A_fem_coeff(x):
    return A(x, theta(x) / eps_plot)


x = ufl.SpatialCoordinate(msh)
coeff = fem.Function(V_ref)
A_expr = fem.Expression(A_fem_coeff(x), V_ref.element.interpolation_points())
coeff.interpolate(A_expr)

func_to_png(
    "diffusion_wavy_inclusion_coefficient",
    V_ref,
    coeff,
    cmap="gray",
    save=True,
    title="$A_\\varepsilon$",
    vmax=None,
)


# %%
def A_y(y):
    return A(None, y)


nx = 30
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
msh_micro = mesh.create_unit_square(MPI.COMM_SELF, nx, nx)

pshmm = PoissonStratifiedHMM(msh, A, f, msh_micro, 1e-5, Dtheta)
pshmm.set_boundary_conditions(darcy_flow_bcs(pshmm.function_space))
u_pshmm = pshmm.solve()

cells, types, x = plot.vtk_mesh(pshmm.function_space)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u_pshmm.x.array
grid.set_active_scalars("u")
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar(factor=1 / np.max(u_ref.x.array))
plotter.add_mesh(warped)

plotter.show()
func_to_png(
    "diffusion_wavy_inclusions_stratified_hmm",
    pshmm.function_space,
    u_pshmm,
    save=True,
    title="$u_\\text{hmm}$",
)


# %%


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


u_pshmm_interp = interpolate_nonmatching(V_ref, pshmm.function_space, u_pshmm)

print(f"relative L^2 error is: {l2_relative_error(u_ref, u_pshmm_interp)}")
