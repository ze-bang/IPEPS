"""
iPEPS State representation.

This module defines the core iPEPS state class that holds the tensor network
representation of a 2D quantum state.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
import h5py
from pathlib import Path

from ipeps.core.tensor import Tensor


@dataclass
class SiteInfo:
    """Information about a single site in the iPEPS."""
    position: Tuple[int, int]  # (x, y) position in unit cell
    sublattice: str  # e.g., 'A', 'B' for honeycomb
    physical_dim: int  # Dimension of physical Hilbert space
    neighbors: Dict[str, Tuple[int, int]]  # Direction -> neighbor position


@dataclass
class BondInfo:
    """Information about a bond in the iPEPS."""
    site1: Tuple[int, int]
    site2: Tuple[int, int]
    bond_dim: int
    direction: str  # e.g., 'horizontal', 'vertical', 'diagonal'


class IPEPSState:
    """
    Infinite Projected Entangled Pair States representation.
    
    An iPEPS is a 2D tensor network state with a finite unit cell that tiles
    the infinite plane. Each site has a rank-5 tensor with one physical index
    and four auxiliary indices connecting to neighbors.
    
    For honeycomb lattice, we use a brick-wall representation with a 2-site
    unit cell, where each tensor has coordination number 3.
    
    Parameters
    ----------
    lattice : object
        Lattice geometry (e.g., HoneycombLattice)
    bond_dim : int
        Bond dimension D for auxiliary indices
    physical_dims : dict or int
        Physical dimension(s) at each site
    chi : int
        Environment bond dimension for CTMRG
    dtype : dtype
        Data type for tensors (default: complex128)
    
    Attributes
    ----------
    tensors : dict
        Dictionary mapping site positions to Tensor objects
    bond_tensors : dict, optional
        Bond tensors (used in simple update)
    environment : object
        CTMRG environment tensors
    
    Examples
    --------
    >>> from ipeps import HoneycombLattice
    >>> lattice = HoneycombLattice(spin_dim=2, boson_dim=3)
    >>> peps = IPEPSState(lattice, bond_dim=4, chi=20)
    >>> print(peps.tensors[(0, 0)].shape)
    """
    
    def __init__(
        self,
        lattice: Any,
        bond_dim: int = 4,
        physical_dims: Optional[Union[int, Dict[Tuple[int, int], int]]] = None,
        chi: int = 20,
        dtype: Any = np.complex128,
        random_init: bool = True,
    ):
        self.lattice = lattice
        self.bond_dim = bond_dim
        self.chi = chi
        self.dtype = dtype
        
        # Set physical dimensions
        if physical_dims is None:
            physical_dims = lattice.physical_dim
        if isinstance(physical_dims, int):
            self._physical_dims = {pos: physical_dims for pos in lattice.sites}
        else:
            self._physical_dims = physical_dims
        
        # Initialize tensors
        self.tensors: Dict[Tuple[int, int], Tensor] = {}
        self.bond_tensors: Dict[Tuple[Tuple[int, int], str], Tensor] = {}
        
        if random_init:
            self._initialize_random()
        else:
            self._initialize_identity()
        
        # Environment will be computed by CTMRG
        self.environment = None
        
        # Convergence tracking
        self._energy_history: List[float] = []
        self._update_count: int = 0
    
    def _initialize_random(self) -> None:
        """Initialize with random tensors."""
        for pos in self.lattice.sites:
            shape = self._get_tensor_shape(pos)
            tensor = Tensor.random(shape, dtype=self.dtype)
            # Normalize
            tensor = tensor.normalize()
            self.tensors[pos] = tensor
        
        # Initialize bond tensors (identity for now)
        for bond in self.lattice.bonds:
            pos1, pos2, direction = bond
            self.bond_tensors[(pos1, direction)] = Tensor.eye(
                self.bond_dim, dtype=self.dtype
            )
    
    def _initialize_identity(self) -> None:
        """Initialize with product state (identity-like tensors)."""
        for pos in self.lattice.sites:
            shape = self._get_tensor_shape(pos)
            phys_dim = self._physical_dims[pos]
            
            # Create tensor that represents |0⟩ state on each site
            data = np.zeros(shape, dtype=self.dtype)
            
            # Set |0⟩ component with minimal entanglement
            # For a 3-coordinated site: shape = (d, D, D, D)
            if len(shape) == 4:
                data[0, 0, 0, 0] = 1.0
            elif len(shape) == 5:
                data[0, 0, 0, 0, 0] = 1.0
            
            self.tensors[pos] = Tensor(data, dtype=self.dtype)
        
        # Initialize bond tensors as identity
        for bond in self.lattice.bonds:
            pos1, pos2, direction = bond
            self.bond_tensors[(pos1, direction)] = Tensor.eye(
                self.bond_dim, dtype=self.dtype
            )
    
    def _get_tensor_shape(self, pos: Tuple[int, int]) -> Tuple[int, ...]:
        """Get the shape of tensor at a given position."""
        phys_dim = self._physical_dims[pos]
        coordination = self.lattice.get_coordination(pos)
        
        # Shape: (physical, aux1, aux2, ..., aux_n)
        return (phys_dim,) + (self.bond_dim,) * coordination
    
    @property
    def num_sites(self) -> int:
        """Number of sites in the unit cell."""
        return len(self.tensors)
    
    @property
    def total_params(self) -> int:
        """Total number of variational parameters."""
        count = 0
        for tensor in self.tensors.values():
            count += tensor.size * 2  # Complex numbers have 2 real params
        return count
    
    def get_tensor(self, pos: Tuple[int, int]) -> Tensor:
        """Get tensor at a position (with periodic wrapping)."""
        wrapped_pos = self.lattice.wrap_position(pos)
        return self.tensors[wrapped_pos]
    
    def set_tensor(self, pos: Tuple[int, int], tensor: Tensor) -> None:
        """Set tensor at a position."""
        wrapped_pos = self.lattice.wrap_position(pos)
        self.tensors[wrapped_pos] = tensor
    
    def get_bond_tensor(self, pos: Tuple[int, int], direction: str) -> Tensor:
        """Get bond tensor (singular values) for simple update."""
        key = (self.lattice.wrap_position(pos), direction)
        if key in self.bond_tensors:
            return self.bond_tensors[key]
        return Tensor.eye(self.bond_dim, dtype=self.dtype)
    
    def set_bond_tensor(self, pos: Tuple[int, int], direction: str, tensor: Tensor) -> None:
        """Set bond tensor."""
        wrapped_pos = self.lattice.wrap_position(pos)
        self.bond_tensors[(wrapped_pos, direction)] = tensor
    
    def absorb_bond_tensors(self) -> 'IPEPSState':
        """
        Absorb bond tensors into site tensors.
        
        Used when transitioning from simple update to full update.
        
        Returns
        -------
        IPEPSState
            New state with absorbed bond tensors
        """
        # Create copy
        new_state = self.copy()
        
        for pos in self.lattice.sites:
            tensor = new_state.tensors[pos]
            
            # For each bond direction, absorb sqrt of bond tensor
            for idx, direction in enumerate(self.lattice.get_directions(pos)):
                bond = new_state.get_bond_tensor(pos, direction)
                sqrt_bond = bond.sqrt()
                
                # Contract with the appropriate axis
                # Axis 0 is physical, so bond axis is idx + 1
                axis = idx + 1
                
                # Contract: tensor_{...i...} * sqrt_bond_{ij} -> tensor_{...j...}
                tensor = self._contract_bond_axis(tensor, sqrt_bond, axis)
            
            new_state.tensors[pos] = tensor
        
        # Clear bond tensors (now identity)
        for key in new_state.bond_tensors:
            new_state.bond_tensors[key] = Tensor.eye(new_state.bond_dim, dtype=self.dtype)
        
        return new_state
    
    def _contract_bond_axis(self, tensor: Tensor, bond: Tensor, axis: int) -> Tensor:
        """Contract a bond tensor into a specific axis of a site tensor."""
        # Move axis to last position
        ndim = tensor.ndim
        axes = list(range(ndim))
        axes.remove(axis)
        axes.append(axis)
        
        tensor = tensor.transpose(tuple(axes))
        
        # Contract: (..., D) @ (D, D) -> (..., D)
        result = tensor.tensordot(bond, axes=([ndim-1], [0]))
        
        # Move axis back
        inverse_perm = [axes.index(i) for i in range(ndim)]
        return result.transpose(tuple(inverse_perm))
    
    def copy(self) -> 'IPEPSState':
        """Create a deep copy of the state."""
        new_state = IPEPSState(
            self.lattice,
            bond_dim=self.bond_dim,
            physical_dims=self._physical_dims.copy(),
            chi=self.chi,
            dtype=self.dtype,
            random_init=False,
        )
        
        # Copy tensors
        for pos, tensor in self.tensors.items():
            new_state.tensors[pos] = tensor.clone()
        
        for key, tensor in self.bond_tensors.items():
            new_state.bond_tensors[key] = tensor.clone()
        
        new_state._energy_history = self._energy_history.copy()
        new_state._update_count = self._update_count
        
        return new_state
    
    def normalize(self) -> None:
        """Normalize the state tensors."""
        for pos in self.tensors:
            self.tensors[pos] = self.tensors[pos].normalize()
    
    def compute_norm(self) -> float:
        """
        Compute the norm ⟨ψ|ψ⟩ using the environment.
        
        Returns
        -------
        float
            The squared norm of the state
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_norm()
    
    def compute_energy(self, hamiltonian: Any) -> float:
        """
        Compute the energy ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            The Hamiltonian operator
        
        Returns
        -------
        float
            The energy per site
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_energy(hamiltonian)
    
    def compute_observable(
        self,
        operator: Union[str, Tensor],
        site: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Compute expectation value of a local operator.
        
        Parameters
        ----------
        operator : str or Tensor
            Operator name ('Sx', 'Sy', 'Sz', 'n', ...) or explicit operator tensor
        site : tuple, optional
            Site position. If None, averages over all sites.
        
        Returns
        -------
        float
            Expectation value
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_observable(operator, site)
    
    def compute_correlator(
        self,
        operator1: Union[str, Tensor],
        operator2: Union[str, Tensor],
        site1: Tuple[int, int],
        site2: Tuple[int, int],
    ) -> complex:
        """
        Compute two-point correlation function.
        
        Parameters
        ----------
        operator1, operator2 : str or Tensor
            Operators at each site
        site1, site2 : tuple
            Site positions
        
        Returns
        -------
        complex
            Correlation function value
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_correlator(operator1, operator2, site1, site2)
    
    def get_reduced_density_matrix(
        self,
        sites: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Compute reduced density matrix for a set of sites.
        
        Parameters
        ----------
        sites : list of tuple
            Sites to include in the reduced density matrix
        
        Returns
        -------
        Tensor
            Reduced density matrix
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_rdm(sites)
    
    def get_entanglement_spectrum(
        self,
        cut_direction: str = 'horizontal',
    ) -> np.ndarray:
        """
        Compute entanglement spectrum for a bipartition.
        
        Parameters
        ----------
        cut_direction : str
            Direction of the entanglement cut
        
        Returns
        -------
        ndarray
            Entanglement spectrum (eigenvalues of reduced density matrix)
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Run CTMRG first.")
        
        return self.environment.compute_entanglement_spectrum(cut_direction)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the iPEPS state to HDF5 file.
        
        Parameters
        ----------
        path : str or Path
            Output file path
        """
        path = Path(path)
        
        with h5py.File(path, 'w') as f:
            # Metadata
            f.attrs['bond_dim'] = self.bond_dim
            f.attrs['chi'] = self.chi
            f.attrs['num_sites'] = self.num_sites
            f.attrs['update_count'] = self._update_count
            
            # Lattice info
            lattice_grp = f.create_group('lattice')
            lattice_grp.attrs['type'] = type(self.lattice).__name__
            
            # Site tensors
            tensors_grp = f.create_group('tensors')
            for pos, tensor in self.tensors.items():
                key = f'{pos[0]}_{pos[1]}'
                tensors_grp.create_dataset(key, data=tensor.numpy())
            
            # Bond tensors
            bonds_grp = f.create_group('bond_tensors')
            for (pos, direction), tensor in self.bond_tensors.items():
                key = f'{pos[0]}_{pos[1]}_{direction}'
                bonds_grp.create_dataset(key, data=tensor.numpy())
            
            # Energy history
            if self._energy_history:
                f.create_dataset('energy_history', data=self._energy_history)
    
    @classmethod
    def load(cls, path: Union[str, Path], lattice: Any) -> 'IPEPSState':
        """
        Load an iPEPS state from HDF5 file.
        
        Parameters
        ----------
        path : str or Path
            Input file path
        lattice : object
            Lattice geometry object
        
        Returns
        -------
        IPEPSState
            Loaded state
        """
        path = Path(path)
        
        with h5py.File(path, 'r') as f:
            bond_dim = f.attrs['bond_dim']
            chi = f.attrs['chi']
            
            state = cls(
                lattice,
                bond_dim=bond_dim,
                chi=chi,
                random_init=False,
            )
            
            # Load tensors
            for key in f['tensors']:
                parts = key.split('_')
                pos = (int(parts[0]), int(parts[1]))
                data = f['tensors'][key][:]
                state.tensors[pos] = Tensor(data)
            
            # Load bond tensors
            if 'bond_tensors' in f:
                for key in f['bond_tensors']:
                    parts = key.split('_')
                    pos = (int(parts[0]), int(parts[1]))
                    direction = parts[2]
                    data = f['bond_tensors'][key][:]
                    state.bond_tensors[(pos, direction)] = Tensor(data)
            
            # Load history
            if 'energy_history' in f:
                state._energy_history = list(f['energy_history'][:])
            
            state._update_count = f.attrs.get('update_count', 0)
        
        return state
    
    def __repr__(self) -> str:
        return (
            f"IPEPSState(sites={self.num_sites}, D={self.bond_dim}, "
            f"χ={self.chi}, params={self.total_params})"
        )
