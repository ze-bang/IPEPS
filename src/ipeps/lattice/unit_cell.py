"""
Unit cell definition for iPEPS lattice geometries.

Provides abstract base class and utilities for defining unit cells
in tensor network representations.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Site:
    """
    Represents a site in the unit cell.
    
    Attributes
    ----------
    position : tuple
        (x, y) position within the unit cell
    sublattice : str
        Sublattice label (e.g., 'A', 'B')
    physical_dim : int
        Dimension of local Hilbert space
    coordination : int
        Number of neighboring sites
    """
    position: Tuple[int, int]
    sublattice: str = 'A'
    physical_dim: int = 2
    coordination: int = 4


@dataclass
class Bond:
    """
    Represents a bond between two sites.
    
    Attributes
    ----------
    site1 : tuple
        Position of first site
    site2 : tuple
        Position of second site
    direction : str
        Direction label (e.g., 'x', 'y', 'z' for honeycomb)
    strength : float
        Bond strength modifier (default 1.0)
    """
    site1: Tuple[int, int]
    site2: Tuple[int, int]
    direction: str
    strength: float = 1.0
    
    def __hash__(self):
        # Canonical ordering for bond hashing
        s1, s2 = sorted([self.site1, self.site2])
        return hash((s1, s2, self.direction))
    
    def __eq__(self, other):
        if not isinstance(other, Bond):
            return False
        s1, s2 = sorted([self.site1, self.site2])
        o1, o2 = sorted([other.site1, other.site2])
        return s1 == o1 and s2 == o2 and self.direction == other.direction


class UnitCell(ABC):
    """
    Abstract base class for unit cell definitions.
    
    Subclasses must implement:
    - sites: List of Site objects
    - bonds: List of Bond objects  
    - wrap_position: Periodic boundary wrapping
    - get_neighbor: Get neighbor in a direction
    
    Parameters
    ----------
    Lx : int
        Unit cell size in x direction
    Ly : int
        Unit cell size in y direction
    physical_dim : int
        Default physical dimension
    """
    
    def __init__(
        self,
        Lx: int = 1,
        Ly: int = 1,
        physical_dim: int = 2,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self._physical_dim = physical_dim
        
        self._sites: Dict[Tuple[int, int], Site] = {}
        self._bonds: List[Bond] = []
        self._bond_set: Set[Bond] = set()
        
        self._initialize_structure()
    
    @abstractmethod
    def _initialize_structure(self) -> None:
        """Initialize sites and bonds. Must be implemented by subclasses."""
        pass
    
    @property
    def sites(self) -> List[Tuple[int, int]]:
        """List of site positions in the unit cell."""
        return list(self._sites.keys())
    
    @property
    def bonds(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """List of bonds as (site1, site2, direction) tuples."""
        return [(b.site1, b.site2, b.direction) for b in self._bonds]
    
    @property
    def num_sites(self) -> int:
        """Number of sites in the unit cell."""
        return len(self._sites)
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds in the unit cell."""
        return len(self._bonds)
    
    @property
    def physical_dim(self) -> int:
        """Default physical dimension."""
        return self._physical_dim
    
    def get_site(self, pos: Tuple[int, int]) -> Site:
        """Get Site object at a position."""
        wrapped = self.wrap_position(pos)
        return self._sites.get(wrapped)
    
    def get_coordination(self, pos: Tuple[int, int]) -> int:
        """Get coordination number at a position."""
        site = self.get_site(pos)
        return site.coordination if site else 0
    
    def get_sublattice(self, pos: Tuple[int, int]) -> str:
        """Get sublattice label at a position."""
        site = self.get_site(pos)
        return site.sublattice if site else 'A'
    
    @abstractmethod
    def wrap_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Wrap position to unit cell with periodic boundaries."""
        pass
    
    @abstractmethod
    def get_neighbor(
        self,
        pos: Tuple[int, int],
        direction: str,
    ) -> Tuple[int, int]:
        """Get neighboring site in given direction."""
        pass
    
    @abstractmethod
    def get_directions(self, pos: Tuple[int, int]) -> List[str]:
        """Get list of bond directions from a site."""
        pass
    
    def get_bonds_at_site(self, pos: Tuple[int, int]) -> List[Bond]:
        """Get all bonds connected to a site."""
        wrapped = self.wrap_position(pos)
        return [b for b in self._bonds if b.site1 == wrapped or b.site2 == wrapped]
    
    def __iter__(self) -> Iterator[Tuple[int, int]]:
        """Iterate over site positions."""
        return iter(self._sites.keys())
    
    def __len__(self) -> int:
        return len(self._sites)
    
    def __contains__(self, pos: Tuple[int, int]) -> bool:
        wrapped = self.wrap_position(pos)
        return wrapped in self._sites


class SquareLattice(UnitCell):
    """
    Square lattice unit cell.
    
    Standard square lattice with 4-fold coordination.
    
    Parameters
    ----------
    Lx, Ly : int
        Unit cell dimensions
    physical_dim : int
        Local Hilbert space dimension
    """
    
    def _initialize_structure(self) -> None:
        """Initialize square lattice structure."""
        # Add sites
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                sublattice = 'A' if (x + y) % 2 == 0 else 'B'
                self._sites[pos] = Site(
                    position=pos,
                    sublattice=sublattice,
                    physical_dim=self._physical_dim,
                    coordination=4,
                )
        
        # Add bonds
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                
                # Right neighbor (x-bond)
                right = self.wrap_position((x + 1, y))
                bond_x = Bond(pos, right, 'x')
                if bond_x not in self._bond_set:
                    self._bonds.append(bond_x)
                    self._bond_set.add(bond_x)
                
                # Up neighbor (y-bond)
                up = self.wrap_position((x, y + 1))
                bond_y = Bond(pos, up, 'y')
                if bond_y not in self._bond_set:
                    self._bonds.append(bond_y)
                    self._bond_set.add(bond_y)
    
    def wrap_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Wrap to unit cell."""
        x, y = pos
        return (x % self.Lx, y % self.Ly)
    
    def get_neighbor(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Get neighbor in direction."""
        x, y = pos
        if direction == 'x' or direction == 'right':
            return self.wrap_position((x + 1, y))
        elif direction == '-x' or direction == 'left':
            return self.wrap_position((x - 1, y))
        elif direction == 'y' or direction == 'up':
            return self.wrap_position((x, y + 1))
        elif direction == '-y' or direction == 'down':
            return self.wrap_position((x, y - 1))
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_directions(self, pos: Tuple[int, int]) -> List[str]:
        """Bond directions from a site."""
        return ['x', 'y', '-x', '-y']
