"""Basic FEM problems together with dolfinx solutions.

Classes:
    PeriodicLinearEquation: general base class for fully periodic problems.
    PeriodicPoisson: solves the Poisson Equation on a fully periodic mesh.
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx_mpc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from hommx.cell_problem import PeriodicLinearProblem
from hommx.helpers import create_periodic_boundary_conditions


class PeriodicLinearEquation:
    r"""Solves a periodic problem on a given mesh.

    The mesh is assumed to be a rectangle, otherwise the master-slave mapping of the periodic DoFs
    may not work.


    $$
    a(u, v) = L(v)
    $$
    with $u(x, y_1) = u(x, y_0)$ and $u(x_1, y) = u(x_0, y)$ for all $x, y$.

    """

    def __init__(
        self,
        msh: mesh.Mesh,
        a: ufl.Form,
        L: ufl.Form,
    ):
        """Initialization of Poisson problem.

        Args:
            msh: Mesh to solve the linear problem on
            a: LHS billinear form
            L: right hand side linear form
        """
        self._msh = msh
        self._a = a
        self._L = L
        self._V = fem.functionspace(self._msh, ("Lagrange", 1))
        self._u = ufl.TrialFunction(self._V)
        self._v = ufl.TestFunction(self._V)
        self._fdim = msh.topology.dim - 1
        self._bcs = []  # Dirichlet Boundary conditions are not currently supported, but we drag along a placeholder
        self._mpc = create_periodic_boundary_conditions(self._msh, self._V, self._bcs)
        self._problem = LinearProblem(self._a, self._L, self._mpc)

    def _set_up_nullspace(self):
        nullspace_vector = PETSc.Vec().create(MPI.COMM_WORLD)
        nullspace_vector.setSizes(self._problem._A.getSize()[0])
        nullspace_vector.setUp()
        nullspace_vector.set(1.0)
        nullspace_vector.setValues(self._mpc.slaves, np.zeros_like(self._mpc.slaves))
        nullspace_vector.assemble()
        nullspace = PETSc.NullSpace().create(vectors=(nullspace_vector,), comm=MPI.COMM_WORLD)
        self._A.setNullSpace(nullspace)
        nullspace.remove(self._b)

    def solve(self) -> fem.Function:
        """Solves the problem and returns the solution."""
        problem = PeriodicLinearProblem(self._a, self._L, self._mpc)
        uh = problem.solve(self._V, self._v, self._msh)
        return uh


class PeriodicPoisson(PeriodicLinearEquation):
    r"""Solves the periodic Poisson equation on a given square mesh.

    $$
        \int_\Omega A \nabla u \cdot \nabla v dx = \int_\Omega f v dx
    $$


    """

    def __init__(
        self,
        msh: mesh.Mesh,
        A: ufl.Form,
        f: ufl.Form,
    ):
        """Initialization of Poisson problem.

        Args:
            msh: The mesh on which the periodic poisson problem is to be solved.
            A (ufl.Form): Coefficient
            f (ufl.Form): Source term
        """
        super().__init__(msh, None, None)

        self._a = ufl.inner(A * ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        self._L = ufl.inner(f, self._v) * ufl.dx
