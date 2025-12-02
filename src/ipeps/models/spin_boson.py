"""
Spin-Boson Hamiltonians for honeycomb lattice.

This module implements various spin-boson coupled Hamiltonians
for studying quantum phase transitions and novel phases of matter.

Supported models:
- General spin-boson coupling (Holstein-type)
- Kitaev model with phonon coupling
- Heisenberg-Holstein model
- Custom Hamiltonians

The Hamiltonians support:
- Multiple coupling schemes
- Bond-dependent interactions
- External fields
- Dissipative bath coupling (via Lindblad operators)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ipeps.core.tensor import Tensor
from ipeps.models.operators import (
    SpinOperators,
    BosonOperators,
    SpinBosonOperators,
)


@dataclass
class HamiltonianParams:
    """
    Container for Hamiltonian parameters.
    
    Parameters
    ----------
    J : float
        Spin exchange coupling (default: 1.0)
    g : float
        Spin-boson coupling strength (default: 0.0)
    omega : float
        Boson frequency (default: 1.0)
    h : float
        External magnetic field (default: 0.0)
    h_direction : str
        Field direction ('x', 'y', 'z') (default: 'z')
    delta : float
        Anisotropy parameter (default: 1.0, isotropic)
    K : float
        Kitaev coupling (default: 0.0)
    Gamma : float
        Symmetric off-diagonal coupling (default: 0.0)
    """
    J: float = 1.0
    g: float = 0.0
    omega: float = 1.0
    h: float = 0.0
    h_direction: str = 'z'
    delta: float = 1.0
    K: float = 0.0
    Gamma: float = 0.0
    
    # Bond-specific couplings (optional)
    J_x: Optional[float] = None
    J_y: Optional[float] = None
    J_z: Optional[float] = None
    
    def get_bond_coupling(self, direction: str) -> float:
        """Get coupling for a specific bond direction."""
        if direction == 'x' and self.J_x is not None:
            return self.J_x
        elif direction == 'y' and self.J_y is not None:
            return self.J_y
        elif direction == 'z' and self.J_z is not None:
            return self.J_z
        return self.J


class Hamiltonian(ABC):
    """
    Abstract base class for Hamiltonians.
    
    Subclasses must implement:
    - get_bond_hamiltonian: Returns the two-site Hamiltonian for a bond
    - get_site_hamiltonian: Returns single-site terms
    """
    
    def __init__(
        self,
        lattice: Any,
        params: Optional[HamiltonianParams] = None,
    ):
        self.lattice = lattice
        self.params = params or HamiltonianParams()
        
        # Build operators
        self._build_operators()
    
    @abstractmethod
    def _build_operators(self) -> None:
        """Build the operators needed for this Hamiltonian."""
        pass
    
    @abstractmethod
    def get_bond_hamiltonian(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
    ) -> Tensor:
        """
        Get the two-site Hamiltonian for a bond.
        
        Parameters
        ----------
        site1, site2 : tuple
            Site positions
        direction : str
            Bond direction
        
        Returns
        -------
        Tensor
            Bond Hamiltonian matrix (d²×d² or d₁d₂×d₁d₂)
        """
        pass
    
    @abstractmethod
    def get_site_hamiltonian(self, site: Tuple[int, int]) -> Tensor:
        """
        Get single-site Hamiltonian terms.
        
        Parameters
        ----------
        site : tuple
            Site position
        
        Returns
        -------
        Tensor
            On-site Hamiltonian matrix (d×d)
        """
        pass
    
    def get_time_evolution_operator(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
        dt: float,
        order: int = 2,
    ) -> Tensor:
        """
        Get imaginary time evolution operator for a bond.
        
        U = exp(-dt * H_bond)
        
        Parameters
        ----------
        site1, site2 : tuple
            Site positions
        direction : str
            Bond direction
        dt : float
            Time step (imaginary time)
        order : int
            Suzuki-Trotter order (2 or 4)
        
        Returns
        -------
        Tensor
            Time evolution operator
        """
        from scipy.linalg import expm
        
        H_bond = self.get_bond_hamiltonian(site1, site2, direction)
        U = expm(-dt * H_bond.numpy())
        
        return Tensor(U)
    
    def get_trotter_gates(
        self,
        dt: float,
        order: int = 2,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str, Tensor]]:
        """
        Get Trotter gates for imaginary time evolution.
        
        Returns list of (site1, site2, direction, gate) tuples.
        
        Parameters
        ----------
        dt : float
            Time step
        order : int
            Trotter order (2 or 4)
        
        Returns
        -------
        list
            List of gate tuples for one Trotter step
        """
        gates = []
        
        if order == 2:
            # Second-order Trotter: e^(-dt*H/2) for each term, then reverse
            half_dt = dt / 2
            
            for bond in self.lattice.bonds:
                site1, site2, direction = bond
                gate = self.get_time_evolution_operator(
                    site1, site2, direction, half_dt
                )
                gates.append((site1, site2, direction, gate))
            
            # Reverse order for second half
            for site1, site2, direction, gate in reversed(gates[:-1]):
                gates.append((site1, site2, direction, gate))
        
        elif order == 4:
            # Fourth-order Trotter decomposition
            p = 1 / (4 - 4**(1/3))
            
            for _ in range(2):
                for bond in self.lattice.bonds:
                    site1, site2, direction = bond
                    gate = self.get_time_evolution_operator(
                        site1, site2, direction, p * dt / 2
                    )
                    gates.append((site1, site2, direction, gate))
            
            # Middle step with different coefficient
            for bond in self.lattice.bonds:
                site1, site2, direction = bond
                gate = self.get_time_evolution_operator(
                    site1, site2, direction, (1 - 4*p) * dt / 2
                )
                gates.append((site1, site2, direction, gate))
            
            for _ in range(2):
                for bond in self.lattice.bonds:
                    site1, site2, direction = bond
                    gate = self.get_time_evolution_operator(
                        site1, site2, direction, p * dt / 2
                    )
                    gates.append((site1, site2, direction, gate))
        
        return gates


class SpinBosonHamiltonian(Hamiltonian):
    """
    General spin-boson Hamiltonian on honeycomb lattice.
    
    H = H_spin + H_boson + H_coupling
    
    H_spin = J Σ_<ij> S_i · S_j + h Σ_i S_i^z
    H_boson = ω Σ_i a†_i a_i
    H_coupling = g Σ_i S_i^z (a_i + a†_i)
    
    Parameters
    ----------
    lattice : HoneycombLattice
        The lattice geometry
    J : float
        Spin exchange coupling (default: 1.0)
    g : float
        Spin-boson coupling (default: 0.5)
    omega : float
        Boson frequency (default: 1.0)
    h : float
        External field (default: 0.0)
    boson_dim : int, optional
        Boson truncation. If None, uses lattice.boson_dim
    
    Examples
    --------
    >>> lattice = HoneycombLattice(spin_dim=2, boson_dim=4)
    >>> H = SpinBosonHamiltonian(lattice, J=1.0, g=0.5, omega=1.0)
    >>> H_bond = H.get_bond_hamiltonian((0,0), (1,0), 'x')
    """
    
    def __init__(
        self,
        lattice: Any,
        J: float = 1.0,
        g: float = 0.5,
        omega: float = 1.0,
        h: float = 0.0,
        boson_dim: Optional[int] = None,
        **kwargs,
    ):
        params = HamiltonianParams(J=J, g=g, omega=omega, h=h, **kwargs)
        
        self.spin_dim = lattice.spin_dim
        self.boson_dim = boson_dim or lattice.boson_dim
        self.physical_dim = self.spin_dim * self.boson_dim
        
        super().__init__(lattice, params)
    
    def _build_operators(self) -> None:
        """Build spin-boson operators."""
        self.ops = SpinBosonOperators(self.spin_dim, self.boson_dim)
        
        # Cache commonly used operators
        self.Sx = self.ops.spin_operator('Sx')
        self.Sy = self.ops.spin_operator('Sy')
        self.Sz = self.ops.spin_operator('Sz')
        
        self.a = self.ops.boson_operator('a')
        self.adag = self.ops.boson_operator('adag')
        self.n_b = self.ops.boson_operator('n')
        
        self.I = self.ops.get_identity()
    
    def get_site_hamiltonian(self, site: Tuple[int, int]) -> Tensor:
        """
        Get on-site Hamiltonian.
        
        H_site = ω a†a + g Sz(a + a†) + h Sz
        """
        # Boson energy
        H = self.params.omega * self.n_b
        
        # Spin-boson coupling
        Sz_x = self.ops.coupled_operator('Sz', 'x')  # Sz ⊗ (a + a†)/√2
        H = H + self.params.g * np.sqrt(2) * Sz_x
        
        # External field
        H = H + self.params.h * self.Sz
        
        return H
    
    def get_bond_hamiltonian(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
    ) -> Tensor:
        """
        Get bond Hamiltonian.
        
        H_bond = J (Sx⊗Sx + Sy⊗Sy + Δ·Sz⊗Sz)
        
        Extended to spin-boson space as:
        H_bond^full = H_bond ⊗ I_boson ⊗ I_boson
        """
        d = self.physical_dim
        J = self.params.get_bond_coupling(direction)
        delta = self.params.delta
        
        # Build spin-only bond Hamiltonian
        Sx = self.ops.spin_ops.Sx.numpy()
        Sy = self.ops.spin_ops.Sy.numpy()
        Sz = self.ops.spin_ops.Sz.numpy()
        I_s = np.eye(self.spin_dim, dtype=np.complex128)
        I_b = np.eye(self.boson_dim, dtype=np.complex128)
        
        # Heisenberg: J (S1·S2)
        SxSx = np.kron(Sx, Sx)
        SySy = np.kron(Sy, Sy)
        SzSz = np.kron(Sz, Sz)
        
        H_spin = J * (SxSx + SySy + delta * SzSz)
        
        # Extend to spin-boson space
        # H_bond acts on (spin1 ⊗ boson1) ⊗ (spin2 ⊗ boson2)
        # = spin1 ⊗ spin2 ⊗ boson1 ⊗ boson2 (need to reorder)
        
        # Full identity on bosons
        I_bb = np.eye(self.boson_dim ** 2, dtype=np.complex128)
        
        # Reorder: (s1 s2) ⊗ (b1 b2) -> (s1 b1) ⊗ (s2 b2)
        # This requires a permutation
        
        # Simple approach: build directly in the correct basis
        d_spin = self.spin_dim
        d_bos = self.boson_dim
        d_full = d_spin * d_bos  # Physical dimension per site
        
        H_full = np.zeros((d_full**2, d_full**2), dtype=np.complex128)
        
        # Fill in the spin part
        for s1 in range(d_spin):
            for s2 in range(d_spin):
                for s1p in range(d_spin):
                    for s2p in range(d_spin):
                        # Spin matrix element
                        h_elem = H_spin[s1 * d_spin + s2, s1p * d_spin + s2p]
                        
                        if abs(h_elem) > 1e-15:
                            # This acts as identity on boson indices
                            for b1 in range(d_bos):
                                for b2 in range(d_bos):
                                    # Combined indices
                                    i1 = s1 * d_bos + b1
                                    i2 = s2 * d_bos + b2
                                    j1 = s1p * d_bos + b1
                                    j2 = s2p * d_bos + b2
                                    
                                    row = i1 * d_full + i2
                                    col = j1 * d_full + j2
                                    H_full[row, col] += h_elem
        
        return Tensor(H_full)
    
    def get_total_hamiltonian_mpo(self) -> Dict[Tuple[int, int], Tensor]:
        """
        Get the Hamiltonian as an MPO representation.
        
        Returns dictionary of MPO tensors for each site.
        """
        # For iPEPS, we typically work with the PEPO representation
        # This is a simplified MPO for chain-like contraction
        raise NotImplementedError("MPO representation not yet implemented")


class HolsteinModel(SpinBosonHamiltonian):
    """
    Holstein model: electrons/spins coupled to local phonons.
    
    H = -t Σ_<ij> (c†_i c_j + h.c.) + U Σ_i n_i↑ n_i↓
        + ω Σ_i b†_i b_i + g Σ_i n_i (b_i + b†_i)
    
    For spin representation (half-filling):
    H = J Σ_<ij> (S+_i S-_j + h.c.) + ω Σ_i a†_i a_i + g Σ_i Sz_i (a_i + a†_i)
    
    Parameters
    ----------
    lattice : HoneycombLattice
        Lattice geometry
    t : float
        Hopping amplitude (mapped to J)
    omega : float
        Phonon frequency
    g : float
        Electron-phonon coupling
    U : float
        On-site Hubbard interaction (mapped to field)
    """
    
    def __init__(
        self,
        lattice: Any,
        t: float = 1.0,
        omega: float = 1.0,
        g: float = 0.5,
        U: float = 0.0,
        boson_dim: Optional[int] = None,
    ):
        # Map Holstein to spin language
        # Hopping t -> exchange J
        # On-site U -> can be absorbed in field term
        
        super().__init__(
            lattice,
            J=t,  # XY exchange from hopping
            g=g,
            omega=omega,
            h=U/4,  # Approximate mapping
            boson_dim=boson_dim,
            delta=0.0,  # XY model, no Ising term
        )
        
        self.t = t
        self.U = U


class KitaevSpinBosonModel(SpinBosonHamiltonian):
    """
    Kitaev honeycomb model with spin-phonon coupling.
    
    H = Σ_<ij>_γ K_γ S^γ_i S^γ_j + ω Σ_i a†_i a_i + g Σ_i S^z_i (a_i + a†_i)
    
    where γ ∈ {x, y, z} is the bond type and K_γ are Kitaev couplings.
    
    Parameters
    ----------
    lattice : HoneycombLattice
        Honeycomb lattice
    K_x, K_y, K_z : float
        Kitaev couplings for each bond type
    omega : float
        Phonon frequency
    g : float
        Spin-phonon coupling
    h : float
        External magnetic field
    """
    
    def __init__(
        self,
        lattice: Any,
        K_x: float = 1.0,
        K_y: float = 1.0,
        K_z: float = 1.0,
        omega: float = 1.0,
        g: float = 0.0,
        h: float = 0.0,
        boson_dim: Optional[int] = None,
    ):
        params = HamiltonianParams(
            J=0.0,  # No Heisenberg term
            K=1.0,  # Enable Kitaev
            J_x=K_x,
            J_y=K_y,
            J_z=K_z,
            omega=omega,
            g=g,
            h=h,
        )
        
        self.K_x = K_x
        self.K_y = K_y
        self.K_z = K_z
        self.spin_dim = lattice.spin_dim
        self.boson_dim = boson_dim or lattice.boson_dim
        self.physical_dim = self.spin_dim * self.boson_dim
        
        Hamiltonian.__init__(self, lattice, params)
    
    def _build_operators(self) -> None:
        """Build operators for Kitaev model."""
        self.ops = SpinBosonOperators(self.spin_dim, self.boson_dim)
        
        self.Sx = self.ops.spin_operator('Sx')
        self.Sy = self.ops.spin_operator('Sy')
        self.Sz = self.ops.spin_operator('Sz')
        
        self.a = self.ops.boson_operator('a')
        self.adag = self.ops.boson_operator('adag')
        self.n_b = self.ops.boson_operator('n')
        
        self.I = self.ops.get_identity()
    
    def get_bond_hamiltonian(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
    ) -> Tensor:
        """
        Get Kitaev bond Hamiltonian.
        
        H_γ = K_γ S^γ_i S^γ_j
        """
        d_spin = self.spin_dim
        d_bos = self.boson_dim
        d_full = d_spin * d_bos
        
        # Select the appropriate spin component based on bond type
        if direction == 'x':
            K = self.K_x
            S = self.ops.spin_ops.Sx.numpy()
        elif direction == 'y':
            K = self.K_y
            S = self.ops.spin_ops.Sy.numpy()
        elif direction == 'z':
            K = self.K_z
            S = self.ops.spin_ops.Sz.numpy()
        else:
            # Default to Heisenberg for unknown bonds
            return super().get_bond_hamiltonian(site1, site2, direction)
        
        # S^γ_1 ⊗ S^γ_2
        H_spin = K * np.kron(S, S)
        
        # Extend to spin-boson space (same as parent class)
        H_full = np.zeros((d_full**2, d_full**2), dtype=np.complex128)
        
        for s1 in range(d_spin):
            for s2 in range(d_spin):
                for s1p in range(d_spin):
                    for s2p in range(d_spin):
                        h_elem = H_spin[s1 * d_spin + s2, s1p * d_spin + s2p]
                        
                        if abs(h_elem) > 1e-15:
                            for b1 in range(d_bos):
                                for b2 in range(d_bos):
                                    i1 = s1 * d_bos + b1
                                    i2 = s2 * d_bos + b2
                                    j1 = s1p * d_bos + b1
                                    j2 = s2p * d_bos + b2
                                    
                                    row = i1 * d_full + i2
                                    col = j1 * d_full + j2
                                    H_full[row, col] += h_elem
        
        return Tensor(H_full)


class HeisenbergSpinBosonModel(SpinBosonHamiltonian):
    """
    Heisenberg model with XXZ anisotropy and spin-boson coupling.
    
    H = J Σ_<ij> [Sx_i Sx_j + Sy_i Sy_j + Δ Sz_i Sz_j]
        + h Σ_i Sz_i
        + ω Σ_i a†_i a_i
        + g Σ_i Sz_i (a_i + a†_i)
    
    Parameters
    ----------
    lattice : HoneycombLattice
        Lattice geometry
    J : float
        Exchange coupling
    delta : float
        XXZ anisotropy (Δ = 1 for isotropic Heisenberg)
    h : float
        External magnetic field along z
    omega : float
        Phonon frequency
    g : float
        Spin-phonon coupling
    """
    
    def __init__(
        self,
        lattice: Any,
        J: float = 1.0,
        delta: float = 1.0,
        h: float = 0.0,
        omega: float = 1.0,
        g: float = 0.0,
        boson_dim: Optional[int] = None,
    ):
        super().__init__(
            lattice,
            J=J,
            g=g,
            omega=omega,
            h=h,
            boson_dim=boson_dim,
            delta=delta,
        )


class LangFirsovTransformedHamiltonian(SpinBosonHamiltonian):
    """
    Lang-Firsov transformed Hamiltonian for strong coupling.
    
    The Lang-Firsov transformation eliminates the linear spin-boson
    coupling at the cost of introducing polaron effects.
    
    H_LF = U†HU where U = exp(g/ω Σ_i Sz_i (a†_i - a_i))
    
    This leads to:
    - Reduced hopping: t → t exp(-g²/ω²)
    - Phonon-mediated spin-spin interaction
    
    Use this for strong coupling (g/ω > 1) where perturbation theory fails.
    """
    
    def __init__(
        self,
        lattice: Any,
        J: float = 1.0,
        g: float = 2.0,
        omega: float = 1.0,
        boson_dim: Optional[int] = None,
    ):
        super().__init__(lattice, J, g, omega, boson_dim=boson_dim)
        
        # Compute polaron reduction factor
        self.polaron_factor = np.exp(-g**2 / (2 * omega**2))
        
        # Effective parameters
        self.J_eff = J * self.polaron_factor
    
    def get_bond_hamiltonian(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
    ) -> Tensor:
        """
        Get Lang-Firsov transformed bond Hamiltonian.
        
        The transformation dresses spin operators with phonon clouds.
        """
        # Get untransformed Hamiltonian
        H = super().get_bond_hamiltonian(site1, site2, direction)
        
        # Apply polaron reduction to spin part
        # This is approximate - full treatment needs X operators
        H = H * self.polaron_factor
        
        return H
