"""
iPEPS: Infinite Projected Entangled Pair States for Honeycomb Lattice with Spin-Boson Coupling

A state-of-the-art HPC implementation for tensor network simulations.
"""

__version__ = "1.0.0"
__author__ = "iPEPS Research Team"

from ipeps.core.tensor import Tensor
from ipeps.core.ipeps_state import IPEPSState
from ipeps.lattice.honeycomb import HoneycombLattice
from ipeps.models.spin_boson import SpinBosonHamiltonian
from ipeps.algorithms.ctmrg import CTMRG
from ipeps.algorithms.simple_update import SimpleUpdate
from ipeps.algorithms.full_update import FullUpdate
from ipeps.algorithms.variational import VariationalOptimizer

__all__ = [
    "Tensor",
    "IPEPSState",
    "HoneycombLattice",
    "SpinBosonHamiltonian",
    "CTMRG",
    "SimpleUpdate",
    "FullUpdate",
    "VariationalOptimizer",
]
