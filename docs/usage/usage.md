# Diffusion Equation
We highly advise checking out the [examples](https://github.com/flxrcz/hommx/tree/main/examples).

We only give a quick introduction with some practical tips here.

We expect that the read is used to FEniCSx and dolfinx.
An excellent tutorial for FEniCSx is found under [jsdokken.com/dolfinx-tutorial/](https://jsdokken.com/dolfinx-tutorial/).

Say we want to use the HMM for the diffusion equation

$$
\begin{gather*}
    -\mathrm{div}\big(A_{\varepsilon}(x) \nabla u_{\varepsilon}(x)\big) &= f \hspace{2em}&& \text{in $\Omega$}\\
    u_{\varepsilon}(x) &= 0 \hspace{2em}&& \text{on $\partial \Omega$} \, .
\end{gather*}
$$

Then this can be done by:

```py
import ufl
from dolfinx import mesh
from mpi4py import MPI
from hommx.hmm import PoissonHMM

def A(x, y):
    return 1 + 0.5*ufl.sin(2*ufl.pi*y[0])

def f(x):
    return 1

mesh_macro = mesh.create_unit_square(MPI.COMM_WORLD, nx=20, ny=20)
mesh_micro = mesh.create_unit_square(MPI.COMM_SELF, nx=20, ny=20)
eps = 2**(-8)
hmm_solver = PoissonHMM(mesh_macro, A, f, mesh_micro, eps)
solution = hmm_solver.solve()
```

Of course you can use another mesh for the macroscopic space.
On the microscopic space for now only periodic unit squares (cubes for 3D) are supported.

!!! important "If you want to use the built-in parallelization it is important that the micro mesh lives on `MPI.COMM_SELF`"

We can set boundary conditions with the [`set_boundary_conditions`][hommx.hmm.BaseHMM.set_boundary_conditions] method.

```py
from dolfinx import default_scalar_type
dofs= ...
bc = fem.dirichletbc(value=default_scalar_type(1), dofs=dofs_left, V=hmm_solver.function_space)
hmm_solver.set_boundary_conditions([bc])
solution = hmm_solver.solve()
```

Or set another right-hand side with the [`set_right_hand_side`][hommx.hmm.BaseHMM.set_right_hand_side] method.

```py
def f(x):
    return 2

hmm_solver.set_right_hand_side(f)
solution = hmm_solver.solve()
```

# Parallelization

The implemented parallelization leverages dolfinx distributed mesh automatically to parallelize the solution of the cell problems.
I.e. the cell problems are only solved for owned triangles (tetrahedra).
This should work out of the box, if the mesh lives on `MPI.COMM_WORLD`.

The cell mesh should live on `MPI.COMM_SELF` to make sure that each process can calculate the solution to its cell problems independently without communication.
The library might not work if the cell mesh does not live on `MPI.COMM_SELF`.

For an introduction to using dolfinx with MPI, we recommend: [scientificcomputing.github.io/mpi-tutorial/notebooks/dolfinx_MPI_tutorial.html](https://scientificcomputing.github.io/mpi-tutorial/notebooks/dolfinx_MPI_tutorial.html)

We also recommend [adios4dolfinx](https://github.com/jorgensd/adios4dolfinx)
to read and write meshes as well as functions in parallel.