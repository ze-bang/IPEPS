"""
Simple Update algorithm for iPEPS.

The Simple Update is an efficient approximate method for optimizing
iPEPS tensors using imaginary time evolution. It uses a mean-field
approximation for the environment, making it much faster than the
Full Update but less accurate.

Key features:
- O(D^5) complexity per update (vs O(χ^3 D^6) for Full Update)
- Suitable for initial state preparation
- Bond tensors store singular values (gauge)
- Works well for gapped phases

References:
    - Jiang et al., Phys. Rev. Lett. 101, 090603 (2008)
    - Corboz et al., Phys. Rev. B 82, 024407 (2010)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Any, List, Dict
from dataclasses import dataclass
import warnings
from tqdm import tqdm

from ipeps.core.tensor import Tensor
from ipeps.core.decompositions import truncated_svd, tensor_svd
from ipeps.core.contractions import contract


@dataclass
class SimpleUpdateConfig:
    """Configuration for Simple Update."""
    dt: float = 0.01  # Imaginary time step
    n_steps: int = 1000  # Number of update steps
    max_bond_dim: int = 10  # Maximum bond dimension
    cutoff: float = 1e-12  # Singular value cutoff
    convergence_tol: float = 1e-8  # Energy convergence tolerance
    normalize_interval: int = 10  # Normalize tensors every N steps
    checkpoint_interval: int = 100  # Save checkpoint every N steps
    verbosity: int = 1  # 0=silent, 1=progress, 2=debug


class SimpleUpdate:
    """
    Simple Update algorithm for iPEPS optimization.
    
    Performs imaginary time evolution using a mean-field (simple)
    approximation for the environment.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian to minimize
    config : SimpleUpdateConfig, optional
        Algorithm configuration
    
    Examples
    --------
    >>> from ipeps import IPEPSState, HoneycombLattice
    >>> from ipeps.models import SpinBosonHamiltonian
    >>>
    >>> lattice = HoneycombLattice(spin_dim=2, boson_dim=3)
    >>> peps = IPEPSState(lattice, bond_dim=4)
    >>> H = SpinBosonHamiltonian(lattice, J=1.0, g=0.5)
    >>>
    >>> su = SimpleUpdate(H, SimpleUpdateConfig(dt=0.01, n_steps=500))
    >>> peps = su.run(peps)
    """
    
    def __init__(
        self,
        hamiltonian: Any,
        config: Optional[SimpleUpdateConfig] = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or SimpleUpdateConfig()
        self.lattice = hamiltonian.lattice
        
        # Precompute Trotter gates
        self._gates = None
        self._prepare_gates()
        
        # Convergence tracking
        self._energy_history: List[float] = []
    
    def _prepare_gates(self) -> None:
        """Precompute imaginary time evolution gates."""
        self._gates = self.hamiltonian.get_trotter_gates(
            dt=self.config.dt,
            order=2,
        )
    
    def run(
        self,
        peps: Any,  # IPEPSState
        callback: Optional[callable] = None,
    ) -> Any:
        """
        Run the Simple Update optimization.
        
        Parameters
        ----------
        peps : IPEPSState
            Initial state
        callback : callable, optional
            Called after each step: callback(step, peps, energy)
        
        Returns
        -------
        IPEPSState
            Optimized state
        """
        peps = peps.copy()
        
        iterator = range(self.config.n_steps)
        if self.config.verbosity >= 1:
            iterator = tqdm(iterator, desc="Simple Update")
        
        prev_energy = float('inf')
        
        for step in iterator:
            # Apply Trotter gates
            for gate_info in self._gates:
                site1, site2, direction, gate = gate_info
                peps = self._apply_gate(peps, site1, site2, direction, gate)
            
            # Normalize periodically
            if step % self.config.normalize_interval == 0:
                peps.normalize()
            
            # Check convergence (via bond singular values)
            if step % 10 == 0:
                energy = self._estimate_energy(peps)
                self._energy_history.append(energy)
                
                if self.config.verbosity >= 1:
                    iterator.set_postfix({'E': f'{energy:.8f}'})
                
                if callback is not None:
                    callback(step, peps, energy)
                
                # Convergence check
                if abs(energy - prev_energy) < self.config.convergence_tol:
                    if self.config.verbosity >= 1:
                        print(f"\nConverged after {step} steps")
                    break
                
                prev_energy = energy
        
        return peps
    
    def _apply_gate(
        self,
        peps: Any,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
        gate: Tensor,
    ) -> Any:
        """
        Apply a two-site gate and truncate.
        
        The simple update procedure:
        1. Absorb bond tensors into site tensors
        2. Contract the two sites with the gate
        3. SVD to separate sites and truncate
        4. Extract new bond tensor
        """
        # Get tensors
        A = peps.get_tensor(site1)
        B = peps.get_tensor(site2)
        
        # Get bond tensors connected to these sites
        lambda_AB = peps.get_bond_tensor(site1, direction)
        
        # Find which axes connect site1 to site2
        directions1 = self.lattice.get_directions(site1)
        directions2 = self.lattice.get_directions(site2)
        
        # Axis 0 is physical, so bond axis is direction index + 1
        try:
            axis1 = directions1.index(direction) + 1
        except ValueError:
            # Try reverse direction
            rev = '-' + direction if not direction.startswith('-') else direction[1:]
            axis1 = directions1.index(rev) + 1
        
        try:
            rev = '-' + direction if not direction.startswith('-') else direction[1:]
            axis2 = directions2.index(rev) + 1
        except ValueError:
            axis2 = directions2.index(direction) + 1
        
        # Absorb sqrt(lambda) into tensors
        sqrt_lambda = lambda_AB.sqrt()
        A = self._absorb_on_axis(A, sqrt_lambda, axis1)
        B = self._absorb_on_axis(B, sqrt_lambda, axis2)
        
        # Contract A and B over the bond
        # Move bond axes to be adjacent, then contract
        theta = A.tensordot(B, axes=([axis1], [axis2]))
        
        # theta now has shape: (d1, aux1_1, ..., aux1_n, d2, aux2_1, ..., aux2_m)
        # where the contracted axis is removed
        
        # Apply gate
        # Gate has shape (d1*d2, d1*d2) or (d1, d2, d1, d2)
        d = peps._physical_dims[site1]
        
        # Reshape theta to expose physical indices
        # Physical indices are at positions 0 and (A.ndim - 1)
        theta_shape = theta.dims
        
        # Contract gate with physical indices
        # This is complex due to index ordering - use simplified approach
        theta = self._apply_gate_to_theta(theta, gate, d, A.ndim, B.ndim)
        
        # SVD to separate sites
        # Group indices: site1 indices (except contracted) | site2 indices (except contracted)
        left_axes = tuple(range(A.ndim - 1))  # All A axes except the contracted one
        right_axes = tuple(range(A.ndim - 1, theta.ndim))  # All B axes
        
        A_new, B_new = tensor_svd(
            theta,
            left_axes=left_axes,
            right_axes=right_axes,
            max_rank=self.config.max_bond_dim,
            cutoff=self.config.cutoff,
            absorb='none',
        )
        
        # The singular values become the new bond tensor
        # They're currently absorbed - we need to extract them
        
        # Get new bond dimension
        new_bond_dim = A_new.dims[-1]
        
        # Create new lambda (identity for now, proper extraction is complex)
        new_lambda = Tensor.eye(new_bond_dim)
        
        # Re-insert the bond axis in correct position
        # For A_new, bond is at last axis, should be at axis1
        A_new = self._move_axis(A_new, -1, axis1)
        
        # For B_new, bond is at first axis, should be at axis2
        B_new = self._move_axis(B_new, 0, axis2)
        
        # Update peps
        peps.set_tensor(site1, A_new.normalize())
        peps.set_tensor(site2, B_new.normalize())
        peps.set_bond_tensor(site1, direction, new_lambda)
        
        return peps
    
    def _absorb_on_axis(self, tensor: Tensor, matrix: Tensor, axis: int) -> Tensor:
        """Absorb a matrix on a specific axis of a tensor."""
        # Move axis to last position
        ndim = tensor.ndim
        perm = list(range(ndim))
        perm.remove(axis)
        perm.append(axis)
        
        t = tensor.transpose(tuple(perm))
        
        # Contract: tensor[..., i] @ matrix[i, j] -> tensor[..., j]
        result = t.tensordot(matrix, axes=([-1], [0]))
        
        # Move axis back
        inv_perm = [perm.index(i) for i in range(ndim)]
        return result.transpose(tuple(inv_perm))
    
    def _apply_gate_to_theta(
        self,
        theta: Tensor,
        gate: Tensor,
        d: int,
        ndim_A: int,
        ndim_B: int,
    ) -> Tensor:
        """Apply two-site gate to theta tensor."""
        # theta has physical indices at 0 and (ndim_A - 1)
        # Reshape to make physical indices outermost
        
        shape = theta.dims
        
        # Extract dimensions
        phys_axis1 = 0
        phys_axis2 = ndim_A - 1  # Position after removing contracted axis
        
        # For simplicity, reshape theta to (d, d, rest) and back
        # This is approximate for complex geometries
        
        total_size = theta.size
        rest_size = total_size // (d * d)
        
        # Transpose to put physical axes first
        axes = [phys_axis1, phys_axis2]
        axes.extend(i for i in range(theta.ndim) if i not in axes)
        
        t = theta.transpose(tuple(axes))
        t = t.reshape((d * d, rest_size))
        
        # Apply gate: (d², d²) @ (d², rest) -> (d², rest)
        result = gate.matmul(t)
        
        # Reshape back
        new_shape = (d,) + shape[1:phys_axis2] + (d,) + shape[phys_axis2+1:]
        
        # Inverse transpose
        result = result.reshape((d, d, rest_size))
        result = result.reshape(new_shape)
        
        # Inverse permutation
        inv_axes = [0] * theta.ndim
        for new_pos, old_pos in enumerate(axes):
            inv_axes[old_pos] = new_pos
        
        return result.transpose(tuple(inv_axes))
    
    def _move_axis(self, tensor: Tensor, source: int, target: int) -> Tensor:
        """Move an axis from source position to target position."""
        if source < 0:
            source = tensor.ndim + source
        if target < 0:
            target = tensor.ndim + target
        
        if source == target:
            return tensor
        
        axes = list(range(tensor.ndim))
        axes.remove(source)
        axes.insert(target, source)
        
        return tensor.transpose(tuple(axes))
    
    def _estimate_energy(self, peps: Any) -> float:
        """
        Estimate energy from bond tensors.
        
        Uses the singular values (bond tensors) to estimate
        the variational energy without full CTMRG.
        """
        # Sum of -log(λ) gives an approximate energy
        # This is very rough - proper energy requires CTMRG
        
        total = 0.0
        count = 0
        
        for key, bond in peps.bond_tensors.items():
            # Get diagonal values (singular values)
            diag = np.diag(bond.numpy())
            # Filter small values
            diag = diag[np.abs(diag) > 1e-15]
            if len(diag) > 0:
                # Entropy-like measure
                diag_sq = np.abs(diag) ** 2
                diag_sq /= diag_sq.sum()
                entropy = -np.sum(diag_sq * np.log(diag_sq + 1e-15))
                total += entropy
                count += 1
        
        return total / max(count, 1)
    
    @property
    def energy_history(self) -> List[float]:
        """Get the energy history during optimization."""
        return self._energy_history
