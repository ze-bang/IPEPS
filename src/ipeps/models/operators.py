"""
Physical operators for spin and boson systems.

This module provides standard operators used in quantum many-body physics:
- Spin operators (Sx, Sy, Sz, S+, S-)
- Boson operators (a, a†, n)
- Combined spin-boson operators
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

from ipeps.core.tensor import Tensor


@dataclass
class SpinOperators:
    """
    Collection of spin operators for a given spin value.
    
    Parameters
    ----------
    S : float
        Spin quantum number (1/2, 1, 3/2, ...)
    
    Attributes
    ----------
    dim : int
        Hilbert space dimension (2S + 1)
    Sx, Sy, Sz : Tensor
        Spin component operators
    Sp, Sm : Tensor
        Raising and lowering operators
    identity : Tensor
        Identity operator
    """
    S: float
    
    def __post_init__(self):
        self.dim = int(2 * self.S + 1)
        self._build_operators()
    
    def _build_operators(self) -> None:
        """Build spin operators."""
        dim = self.dim
        S = self.S
        
        # m values: S, S-1, ..., -S
        m_vals = np.arange(S, -S - 1, -1)
        
        # S+ operator (raising)
        Sp = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim - 1):
            m = m_vals[i + 1]  # m value of lower state
            Sp[i, i + 1] = np.sqrt(S * (S + 1) - m * (m + 1))
        
        # S- operator (lowering)
        Sm = Sp.T.conj()
        
        # Sx = (S+ + S-) / 2
        Sx = (Sp + Sm) / 2
        
        # Sy = (S+ - S-) / (2i)
        Sy = (Sp - Sm) / (2j)
        
        # Sz operator (diagonal)
        Sz = np.diag(m_vals.astype(np.complex128))
        
        # Identity
        identity = np.eye(dim, dtype=np.complex128)
        
        self.Sx = Tensor(Sx)
        self.Sy = Tensor(Sy)
        self.Sz = Tensor(Sz)
        self.Sp = Tensor(Sp)
        self.Sm = Tensor(Sm)
        self.identity = Tensor(identity)
        
        # Also store S^2 = S(S+1)
        self.S2 = Tensor(S * (S + 1) * identity)
    
    def get(self, name: str) -> Tensor:
        """Get operator by name."""
        operators = {
            'Sx': self.Sx,
            'Sy': self.Sy,
            'Sz': self.Sz,
            'S+': self.Sp,
            'Sp': self.Sp,
            'S-': self.Sm,
            'Sm': self.Sm,
            'I': self.identity,
            'id': self.identity,
            'S2': self.S2,
        }
        if name not in operators:
            raise ValueError(f"Unknown spin operator: {name}")
        return operators[name]
    
    def heisenberg_bond(self, J: float = 1.0) -> Tensor:
        """
        Create Heisenberg bond operator: J * (Sx⊗Sx + Sy⊗Sy + Sz⊗Sz)
        
        Parameters
        ----------
        J : float
            Exchange coupling
        
        Returns
        -------
        Tensor
            Bond Hamiltonian of shape (dim², dim²)
        """
        d = self.dim
        
        # Build as 4-index tensor first, then reshape
        SxSx = np.kron(self.Sx.numpy(), self.Sx.numpy())
        SySy = np.kron(self.Sy.numpy(), self.Sy.numpy())
        SzSz = np.kron(self.Sz.numpy(), self.Sz.numpy())
        
        H_bond = J * (SxSx + SySy + SzSz)
        
        return Tensor(H_bond)
    
    def ising_bond(self, J: float = 1.0) -> Tensor:
        """Create Ising bond operator: J * Sz⊗Sz"""
        SzSz = np.kron(self.Sz.numpy(), self.Sz.numpy())
        return Tensor(J * SzSz)
    
    def xy_bond(self, J: float = 1.0) -> Tensor:
        """Create XY bond operator: J * (Sx⊗Sx + Sy⊗Sy)"""
        SxSx = np.kron(self.Sx.numpy(), self.Sx.numpy())
        SySy = np.kron(self.Sy.numpy(), self.Sy.numpy())
        return Tensor(J * (SxSx + SySy))


@dataclass
class BosonOperators:
    """
    Collection of bosonic operators with truncated Hilbert space.
    
    Parameters
    ----------
    n_max : int
        Maximum boson number (Hilbert space dimension)
    
    Attributes
    ----------
    dim : int
        Hilbert space dimension (n_max)
    a : Tensor
        Annihilation operator
    adag : Tensor
        Creation operator
    n : Tensor
        Number operator
    x : Tensor
        Position operator (a + a†)/√2
    p : Tensor  
        Momentum operator (a - a†)/(i√2)
    """
    n_max: int
    
    def __post_init__(self):
        self.dim = self.n_max
        self._build_operators()
    
    def _build_operators(self) -> None:
        """Build boson operators."""
        dim = self.dim
        
        # Annihilation operator a
        a = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(1, dim):
            a[n - 1, n] = np.sqrt(n)
        
        # Creation operator a†
        adag = a.T.conj()
        
        # Number operator n = a† a
        n_op = np.diag(np.arange(dim, dtype=np.complex128))
        
        # Position operator x = (a + a†) / √2
        x = (a + adag) / np.sqrt(2)
        
        # Momentum operator p = (a - a†) / (i√2)
        p = (a - adag) / (1j * np.sqrt(2))
        
        # Identity
        identity = np.eye(dim, dtype=np.complex128)
        
        self.a = Tensor(a)
        self.adag = Tensor(adag)
        self.n = Tensor(n_op)
        self.x = Tensor(x)
        self.p = Tensor(p)
        self.identity = Tensor(identity)
    
    def get(self, name: str) -> Tensor:
        """Get operator by name."""
        operators = {
            'a': self.a,
            'adag': self.adag,
            'a+': self.adag,
            'a†': self.adag,
            'n': self.n,
            'x': self.x,
            'p': self.p,
            'I': self.identity,
            'id': self.identity,
        }
        if name not in operators:
            raise ValueError(f"Unknown boson operator: {name}")
        return operators[name]
    
    def displacement(self, alpha: complex) -> Tensor:
        """
        Displacement operator D(α) = exp(α a† - α* a)
        
        Parameters
        ----------
        alpha : complex
            Displacement amplitude
        
        Returns
        -------
        Tensor
            Displacement operator matrix
        """
        from scipy.linalg import expm
        
        arg = alpha * self.adag.numpy() - np.conj(alpha) * self.a.numpy()
        D = expm(arg)
        
        return Tensor(D)
    
    def squeeze(self, r: float, phi: float = 0) -> Tensor:
        """
        Squeeze operator S(ξ) = exp((ξ* a² - ξ a†²)/2)
        
        Parameters
        ----------
        r : float
            Squeezing parameter
        phi : float
            Squeezing angle
        
        Returns
        -------
        Tensor
            Squeeze operator matrix
        """
        from scipy.linalg import expm
        
        xi = r * np.exp(1j * phi)
        a2 = self.a.numpy() @ self.a.numpy()
        adag2 = self.adag.numpy() @ self.adag.numpy()
        
        arg = (np.conj(xi) * a2 - xi * adag2) / 2
        S = expm(arg)
        
        return Tensor(S)


class SpinBosonOperators:
    """
    Combined spin-boson operators.
    
    Builds operators acting on the tensor product Hilbert space
    of spin and boson degrees of freedom.
    
    Parameters
    ----------
    spin_dim : int
        Spin Hilbert space dimension
    boson_dim : int
        Boson Hilbert space dimension (n_max)
    S : float, optional
        Spin quantum number. If not given, inferred from spin_dim.
    """
    
    def __init__(
        self,
        spin_dim: int,
        boson_dim: int,
        S: Optional[float] = None,
    ):
        self.spin_dim = spin_dim
        self.boson_dim = boson_dim
        self.dim = spin_dim * boson_dim
        
        # Infer spin from dimension
        if S is None:
            S = (spin_dim - 1) / 2
        
        self.spin_ops = SpinOperators(S)
        self.boson_ops = BosonOperators(boson_dim)
        
        # Identity operators for tensor products
        self._spin_id = np.eye(spin_dim, dtype=np.complex128)
        self._boson_id = np.eye(boson_dim, dtype=np.complex128)
    
    def spin_operator(self, name: str) -> Tensor:
        """Get spin operator extended to full Hilbert space."""
        op = self.spin_ops.get(name).numpy()
        # op ⊗ I_boson
        full_op = np.kron(op, self._boson_id)
        return Tensor(full_op)
    
    def boson_operator(self, name: str) -> Tensor:
        """Get boson operator extended to full Hilbert space."""
        op = self.boson_ops.get(name).numpy()
        # I_spin ⊗ op
        full_op = np.kron(self._spin_id, op)
        return Tensor(full_op)
    
    def coupled_operator(
        self,
        spin_op: str,
        boson_op: str,
    ) -> Tensor:
        """
        Create coupled spin-boson operator.
        
        Parameters
        ----------
        spin_op : str
            Name of spin operator
        boson_op : str
            Name of boson operator
        
        Returns
        -------
        Tensor
            Product operator acting on full Hilbert space
        """
        s_op = self.spin_ops.get(spin_op).numpy()
        b_op = self.boson_ops.get(boson_op).numpy()
        
        full_op = np.kron(s_op, b_op)
        return Tensor(full_op)
    
    def get_identity(self) -> Tensor:
        """Get identity on full Hilbert space."""
        return Tensor(np.eye(self.dim, dtype=np.complex128))


def spin_operators(S: float = 0.5) -> SpinOperators:
    """
    Create spin operators for given spin.
    
    Parameters
    ----------
    S : float
        Spin quantum number (default: 1/2)
    
    Returns
    -------
    SpinOperators
        Collection of spin operators
    """
    return SpinOperators(S)


def boson_operators(n_max: int = 5) -> BosonOperators:
    """
    Create boson operators with truncation.
    
    Parameters
    ----------
    n_max : int
        Maximum boson number
    
    Returns
    -------
    BosonOperators
        Collection of boson operators
    """
    return BosonOperators(n_max)


def get_operator(name: str, dim: int) -> Tensor:
    """
    Get a standard operator by name.
    
    Parameters
    ----------
    name : str
        Operator name ('Sx', 'Sy', 'Sz', 'Sp', 'Sm', 'n', 'a', 'adag', ...)
    dim : int
        Hilbert space dimension
    
    Returns
    -------
    Tensor
        The requested operator
    """
    # Check if it's a spin operator
    spin_names = {'Sx', 'Sy', 'Sz', 'Sp', 'Sm', 'S+', 'S-', 'S2'}
    boson_names = {'a', 'adag', 'a+', 'a†', 'n', 'x', 'p'}
    
    if name in spin_names:
        S = (dim - 1) / 2
        ops = SpinOperators(S)
        return ops.get(name)
    
    elif name in boson_names:
        ops = BosonOperators(dim)
        return ops.get(name)
    
    elif name == 'I' or name == 'id' or name == 'identity':
        return Tensor(np.eye(dim, dtype=np.complex128))
    
    else:
        raise ValueError(f"Unknown operator: {name}")


def pauli_matrices() -> Dict[str, Tensor]:
    """
    Get Pauli matrices.
    
    Returns
    -------
    dict
        Dictionary with 'X', 'Y', 'Z', 'I' keys
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    
    return {
        'X': Tensor(sigma_x),
        'Y': Tensor(sigma_y),
        'Z': Tensor(sigma_z),
        'I': Tensor(identity),
    }


def gell_mann_matrices() -> Dict[str, Tensor]:
    """
    Get Gell-Mann matrices (generators of SU(3)).
    
    Useful for spin-1 systems.
    
    Returns
    -------
    dict
        Dictionary with lambda_1 through lambda_8 and identity
    """
    # Standard Gell-Mann matrices
    l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    l8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128) / np.sqrt(3)
    
    return {
        'lambda_1': Tensor(l1),
        'lambda_2': Tensor(l2),
        'lambda_3': Tensor(l3),
        'lambda_4': Tensor(l4),
        'lambda_5': Tensor(l5),
        'lambda_6': Tensor(l6),
        'lambda_7': Tensor(l7),
        'lambda_8': Tensor(l8),
        'I': Tensor(np.eye(3, dtype=np.complex128)),
    }
