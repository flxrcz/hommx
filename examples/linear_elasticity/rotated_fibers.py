# %%
import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot
from mpi4py import MPI

from hommx.hmm import LinearElasticityStratifiedHMM

L = 1
W = 0.4
H = 0.1
delta = W / L
_lambda_ = 1
g = 0.05 * delta**2


# %%
def epsilon(u):
    return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def ufl_circle_indicator(x, y):
    # 1-periodic "wrapped distance" around 1/2 in each coordinate.
    # Using 2*pi here (instead of pi) makes the pattern repeat every 1.
    dx = ufl.acos(ufl.cos(2 * ufl.pi * (x - 1 / 2)))
    dy = ufl.acos(ufl.cos(2 * ufl.pi * (y - 1 / 2)))
    # Keep the same physical radius r=1/4: dx≈2*pi*|x-1/2| near the center.
    return (dx**2 + dy**2) < ((2 * ufl.pi) ** 2 * (1 / 16))


# indicator on circle on y[1]-y[2] plane/fiber along y[0]
def mu(x, y):
    return ufl.conditional(ufl_circle_indicator(y[1], y[2]), 100, 0.001)


def lambda_(y):
    return _lambda_


def theta(x):
    x_0 = x[0]
    x_1 = x[1]
    gamma = 1 / 2 * np.pi * x[1] / W  # from longitudional to vertical in bar from left to right
    x_2 = ufl.cos(gamma) * x[2] - ufl.sin(gamma) * x[0]
    return ufl.as_vector([x_0, x_1, x_2])  # x_0 is unused placeholder


def Dtheta(x):
    """Jacobian of theta(x) as a 3x3 UFL matrix.

    Returns Dtheta_ij = ∂theta_i/∂x_j.
    """

    gamma = 1 / 2 * ufl.pi * x[1] / W

    Dtheta = ufl.as_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, -ufl.sin(gamma), ufl.cos(gamma)],
        ]
    )
    return ufl.transpose(Dtheta)


def A_tensor(x, y):
    """Elasticity tensor A such that A:ε(u) gives stress."""
    I = ufl.Identity(3)
    i, j, k, l = ufl.indices(4)

    # Build explicitly: A_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
    A = ufl.as_tensor(
        lambda_(y) * I[i, j] * I[k, l] + mu(x, y) * (I[i, k] * I[j, l] + I[i, l] * I[j, k]),
        indices=(i, j, k, l),
    )
    return A


# %%
# eps should not play a role here
eps = 2 ** (-5)
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, W, H])],
    [20, 6, 6],
    cell_type=mesh.CellType.tetrahedron,
)
f = fem.Constant(domain, default_scalar_type((0, 0, -g)))
cell_box = mesh.create_unit_cube(MPI.COMM_SELF, 4, 4, 4)
lehmm = LinearElasticityStratifiedHMM(
    domain,
    A_tensor,
    f,
    cell_box,
    eps,
    Dtheta,
    petsc_options_cell_problem={"ksp_atol": 1e-9},
    petsc_options_global_solve={"ksp_type": "cg", "pc_type": "gamg"},
)


def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(
    u_D,
    fem.locate_dofs_topological(lehmm.function_space, fdim, boundary_facets),
    lehmm.function_space,
)
lehmm.set_boundary_conditions(bc)

u_lehmm = lehmm.solve()


# %% Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(lehmm._V_macro)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = u_lehmm.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")

# %%
