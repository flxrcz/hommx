# HOMMX: Heterogeneous Multi-Scale Method with DOLFINx

A Python package implementing the **Heterogeneous Multi-Scale Method (HMM)** for solving multi-scale PDEs using DOLFINx and PETSc.



## Overview

HOMMX provides solvers for multi-scale problems, in particular for the Poisson equation with rapidly oscillating coefficients. The package implements the HMM to efficiently compute homogenized solutions without fully resolving the microscopic scales.

$$
\begin{equation}
    \mathrm{div}(A_\varepsilon\nabla u) = f
\end{equation}
$$

where $A_\varepsilon(x) = A(x, x/\varepsilon)$.

Additionally it includes an implementation for a stratified periodic coefficient.

### Key Features

- **PoissonHMM**: HMM solver for Poisson problems with periodic micro-structure
- **PoissonSemiHMM**: HMM solver for Poisson problems with stratified periodic micro-structure
- Support for 2D and 3D domains
- Integration with DOLFINx's modern FEM framework

## Installation
hommx is available as a conda package on [Prefix]((https://prefix.dev/channels/flxrcz-forge/hommx)).
To install it simply run

```bash
conda install -c https://repo.prefix.dev/flxrcz-forge hommx
```

## Getting started

Have a look at the example files at [examples](https://github.com/flxrcz/hommx/blob/main/examples).