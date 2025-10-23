"""PETSc helpers, mostly for debugging."""

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc


def petsc_matrix_to_numpy(mat: PETSc.Mat):
    mat.assemble()
    if mat.getType() == "nest":
        nr, nc = mat.getNestSize()
        blocks = [[mat.getNestSubMatrix(i, j) for j in range(nc)] for i in range(nr)]
        # Recursively convert submatrices to numpy
        block_arrays = [[petsc_matrix_to_numpy(b) for b in row] for row in blocks]
        return np.block(block_arrays)  # stitch into a dense numpy array
    else:
        ia, ja, av = mat.getValuesCSR()
        nrows, ncols = mat.getSize()
        A_csr = sp.csr_matrix((av, ja, ia), shape=(nrows, ncols))
        return A_csr.toarray()


def petsc_vector_to_numpy(vec: PETSc.Vec):
    return vec.getArray().copy()
