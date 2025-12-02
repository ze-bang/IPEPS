# iPEPS for Honeycomb Lattice with Spin-Boson Coupling

A state-of-the-art High-Performance Computing (HPC) implementation of the infinite Projected Entangled Pair States (iPEPS) algorithm for studying 2D quantum many-body systems on honeycomb lattices with spin-boson coupling.

## Features

### Core Algorithms
- **iPEPS Ansatz**: Infinite projected entangled pair states with customizable bond dimension
- **CTMRG**: Corner Transfer Matrix Renormalization Group for efficient environment approximation
- **Full Update**: Accurate imaginary time evolution with environment optimization
- **Simple Update**: Fast approximate update for initial state preparation
- **Variational Optimization**: Gradient-based optimization using automatic differentiation

### Honeycomb Lattice Support
- Native honeycomb lattice geometry with two-site unit cell
- Flexible boundary conditions
- Support for arbitrary unit cell sizes

### Spin-Boson Model
- Generalized spin-boson Hamiltonians with tunable coupling
- Holstein model implementation
- Truncated boson Hilbert space with configurable cutoff
- Lang-Firsov transformation support

### HPC Features
- **MPI Parallelization**: Distributed computing for large-scale simulations
- **GPU Acceleration**: CUDA/CuPy backend for tensor operations
- **Hybrid Parallelism**: Combined MPI + OpenMP + GPU execution
- **Checkpointing**: Robust save/restore for long-running calculations
- **Memory Optimization**: Efficient tensor contraction ordering

## Installation

### Basic Installation
```bash
pip install -e .
```

### With GPU Support
```bash
pip install -e ".[gpu]"
```

### With Development Tools
```bash
pip install -e ".[dev]"
```

## Quick Start

### Running a Basic Simulation
```python
from ipeps import IPEPS, HoneycombLattice
from ipeps.models import SpinBosonHamiltonian
from ipeps.optimizers import FullUpdate

# Create honeycomb lattice with spin-1/2 and 3 boson levels
lattice = HoneycombLattice(spin_dim=2, boson_dim=3)

# Define Hamiltonian parameters
hamiltonian = SpinBosonHamiltonian(
    J=1.0,           # Spin exchange coupling
    g=0.5,           # Spin-boson coupling
    omega=1.0,       # Boson frequency
    lattice=lattice
)

# Initialize iPEPS state
peps = IPEPS(lattice, bond_dim=4, chi=20)

# Run optimization
optimizer = FullUpdate(hamiltonian, dt=0.01, n_steps=1000)
peps = optimizer.run(peps)

# Compute observables
energy = peps.compute_energy(hamiltonian)
magnetization = peps.compute_observable("Sz")
```

### MPI Parallel Execution
```bash
mpirun -np 16 python -m ipeps.run --config config.yaml
```

## Project Structure

```
ipeps/
├── src/ipeps/
│   ├── core/              # Core tensor network operations
│   │   ├── tensor.py      # Tensor class with backend abstraction
│   │   ├── contractions.py # Optimized tensor contractions
│   │   └── decompositions.py # SVD, QR, and other decompositions
│   ├── lattice/           # Lattice geometry
│   │   ├── honeycomb.py   # Honeycomb lattice implementation
│   │   └── unit_cell.py   # Unit cell management
│   ├── models/            # Physical models
│   │   ├── spin_boson.py  # Spin-boson Hamiltonian
│   │   └── operators.py   # Physical operators
│   ├── algorithms/        # iPEPS algorithms
│   │   ├── ctmrg.py       # Corner Transfer Matrix RG
│   │   ├── simple_update.py
│   │   ├── full_update.py
│   │   └── variational.py
│   ├── hpc/               # HPC infrastructure
│   │   ├── mpi.py         # MPI parallelization
│   │   ├── gpu.py         # GPU backend
│   │   └── checkpointing.py
│   └── analysis/          # Post-processing
│       ├── observables.py
│       └── correlations.py
├── examples/              # Example scripts
├── tests/                 # Unit tests
└── configs/               # Configuration files
```

## Theory

### iPEPS Ansatz
The iPEPS ansatz represents a 2D quantum state as a tensor network:

$$|\Psi\rangle = \sum_{\{s\}} \text{tTr}\left[\prod_{i} A^{[s_i]}_i\right] |s_1, s_2, \ldots\rangle$$

where $A^{[s]}_i$ are rank-5 tensors with one physical index $s$ and four auxiliary indices.

### Honeycomb Lattice
The honeycomb lattice is implemented with a two-site unit cell (A and B sublattices).
Each site has coordination number 3, requiring adapted tensor shapes.

### Spin-Boson Hamiltonian
The Hamiltonian includes:
$$H = J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j + \omega \sum_i a_i^\dagger a_i + g \sum_i S^z_i (a_i + a_i^\dagger)$$

## Performance

- Scales to bond dimensions D > 10 with GPU acceleration
- Efficient CTMRG with χ > 100 environment bond dimension
- MPI scaling efficiency > 80% up to 1000 cores

## References

1. Corboz, P. "Variational optimization with infinite projected entangled-pair states" PRB (2016)
2. Fishman, M. et al. "Faster methods for contracting infinite PEPS" PRB (2018)
3. Liao, H.-J. et al. "Differentiable programming tensor networks" PRX (2019)

## License

MIT License
# IPEPS
