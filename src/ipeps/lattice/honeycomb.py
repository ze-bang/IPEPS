"""
Honeycomb lattice implementation for iPEPS.

The honeycomb lattice is a bipartite lattice with coordination number 3.
We implement it using a brick-wall representation that maps naturally
to a rectangular tensor network.

The unit cell contains two sites (A and B sublattices), and each site
has three neighbors connected by bonds in the x, y, and z directions
(using Kitaev model conventions).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ipeps.lattice.unit_cell import UnitCell, Site, Bond


class HoneycombLattice(UnitCell):
    """
    Honeycomb lattice for iPEPS.
    
    Uses a brick-wall representation with a 2-site unit cell.
    The tensor network representation adapts the coordination number 3
    to fit in a rectangular grid.
    
    Sublattices:
        A sites at (2n, m) positions
        B sites at (2n+1, m) positions
    
    Bond directions (Kitaev convention):
        x-bond: Horizontal bonds (A-B pairs)
        y-bond: Vertical bonds on even columns
        z-bond: Vertical bonds on odd columns (with shift)
    
    Parameters
    ----------
    spin_dim : int
        Dimension of spin Hilbert space (default: 2 for spin-1/2)
    boson_dim : int
        Dimension of boson Hilbert space (default: 1, no bosons)
        If > 1, creates combined spin-boson Hilbert space
    Lx : int
        Number of unit cells in x direction (default: 1)
    Ly : int
        Number of unit cells in y direction (default: 1)
    
    Attributes
    ----------
    spin_dim : int
        Spin Hilbert space dimension
    boson_dim : int
        Boson Hilbert space dimension
    physical_dim : int
        Total physical dimension (spin_dim * boson_dim)
    
    Examples
    --------
    >>> lattice = HoneycombLattice(spin_dim=2, boson_dim=3)
    >>> lattice.physical_dim
    6
    >>> lattice.sites
    [(0, 0), (1, 0)]
    >>> lattice.bonds
    [((0, 0), (1, 0), 'x'), ...]
    """
    
    def __init__(
        self,
        spin_dim: int = 2,
        boson_dim: int = 1,
        Lx: int = 1,
        Ly: int = 1,
    ):
        self.spin_dim = spin_dim
        self.boson_dim = boson_dim
        
        # Total physical dimension is product of spin and boson
        physical_dim = spin_dim * boson_dim
        
        # Use 2*Lx for x direction since we have 2 sites per unit cell
        super().__init__(Lx=2*Lx, Ly=Ly, physical_dim=physical_dim)
    
    def _initialize_structure(self) -> None:
        """Initialize honeycomb lattice structure."""
        # Add sites - alternating A and B sublattices
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                sublattice = 'A' if x % 2 == 0 else 'B'
                
                self._sites[pos] = Site(
                    position=pos,
                    sublattice=sublattice,
                    physical_dim=self._physical_dim,
                    coordination=3,  # Honeycomb has coordination 3
                )
        
        # Add bonds according to honeycomb geometry
        self._initialize_bonds()
    
    def _initialize_bonds(self) -> None:
        """Initialize honeycomb bonds with proper directions."""
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                sublattice = 'A' if x % 2 == 0 else 'B'
                
                if sublattice == 'A':
                    # A sites (even x)
                    # x-bond: connect to B site on the right
                    neighbor_x = self.wrap_position((x + 1, y))
                    self._add_bond(pos, neighbor_x, 'x')
                    
                    # y-bond: connect upward
                    neighbor_y = self.wrap_position((x, y + 1))
                    self._add_bond(pos, neighbor_y, 'y')
                    
                else:
                    # B sites (odd x)
                    # z-bond: diagonal connection
                    # In brick-wall representation, this goes to next row
                    neighbor_z = self.wrap_position((x + 1, y))
                    self._add_bond(pos, neighbor_z, 'z')
    
    def _add_bond(
        self,
        site1: Tuple[int, int],
        site2: Tuple[int, int],
        direction: str,
    ) -> None:
        """Add a bond if it doesn't already exist."""
        bond = Bond(site1, site2, direction)
        if bond not in self._bond_set:
            self._bonds.append(bond)
            self._bond_set.add(bond)
    
    def wrap_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Wrap position to unit cell with periodic boundaries."""
        x, y = pos
        return (x % self.Lx, y % self.Ly)
    
    def get_neighbor(
        self,
        pos: Tuple[int, int],
        direction: str,
    ) -> Tuple[int, int]:
        """
        Get neighboring site in given direction.
        
        Parameters
        ----------
        pos : tuple
            Site position
        direction : str
            Bond direction ('x', 'y', 'z', or with '-' prefix for reverse)
        
        Returns
        -------
        tuple
            Neighbor position
        """
        x, y = pos
        sublattice = 'A' if x % 2 == 0 else 'B'
        
        if direction == 'x':
            if sublattice == 'A':
                return self.wrap_position((x + 1, y))
            else:
                return self.wrap_position((x - 1, y))
        
        elif direction == '-x':
            if sublattice == 'A':
                return self.wrap_position((x - 1, y))
            else:
                return self.wrap_position((x + 1, y))
        
        elif direction == 'y':
            return self.wrap_position((x, y + 1))
        
        elif direction == '-y':
            return self.wrap_position((x, y - 1))
        
        elif direction == 'z':
            if sublattice == 'B':
                return self.wrap_position((x + 1, y))
            else:
                return self.wrap_position((x - 1, y))
        
        elif direction == '-z':
            if sublattice == 'B':
                return self.wrap_position((x - 1, y))
            else:
                return self.wrap_position((x + 1, y))
        
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_directions(self, pos: Tuple[int, int]) -> List[str]:
        """
        Get list of bond directions from a site.
        
        For honeycomb, each site has 3 bonds, but the directions
        depend on the sublattice.
        
        Parameters
        ----------
        pos : tuple
            Site position
        
        Returns
        -------
        list of str
            List of direction labels
        """
        x, y = pos
        sublattice = 'A' if x % 2 == 0 else 'B'
        
        if sublattice == 'A':
            return ['x', 'y', '-y']  # Right, up, down
        else:
            return ['-x', 'z', '-z']  # Left, diagonal bonds
    
    def get_bond_operator_indices(
        self,
        pos: Tuple[int, int],
        direction: str,
    ) -> Tuple[int, int]:
        """
        Get the tensor indices corresponding to a bond.
        
        For applying bond operators in updates.
        
        Parameters
        ----------
        pos : tuple
            Site position
        direction : str
            Bond direction
        
        Returns
        -------
        tuple
            (index in pos tensor, index in neighbor tensor)
        """
        directions = self.get_directions(pos)
        neighbor = self.get_neighbor(pos, direction)
        neighbor_directions = self.get_directions(neighbor)
        
        # Find which auxiliary index corresponds to this direction
        try:
            idx1 = directions.index(direction)
        except ValueError:
            # Try reverse direction
            reverse_dir = '-' + direction if not direction.startswith('-') else direction[1:]
            idx1 = directions.index(reverse_dir)
        
        # For neighbor, find the reverse direction
        reverse_dir = '-' + direction if not direction.startswith('-') else direction[1:]
        try:
            idx2 = neighbor_directions.index(reverse_dir)
        except ValueError:
            idx2 = neighbor_directions.index(direction)
        
        # Auxiliary indices start at 1 (0 is physical)
        return idx1 + 1, idx2 + 1
    
    def get_all_bond_directions(self) -> List[str]:
        """Get all unique bond directions in the lattice."""
        return ['x', 'y', 'z']
    
    def get_plaquette_sites(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the 6 sites forming a hexagonal plaquette.
        
        Parameters
        ----------
        pos : tuple
            Position of one vertex of the plaquette
        
        Returns
        -------
        list of tuple
            Positions of the 6 sites in order around the hexagon
        """
        x, y = pos
        
        # For honeycomb, a plaquette is a hexagon
        # Starting from an A site, go around the hexagon
        if x % 2 == 0:  # A site
            sites = [
                (x, y),           # A
                (x + 1, y),       # B (x-bond)
                (x + 2, y),       # A (z-bond from B)
                (x + 2, y + 1),   # B (y-bond)
                (x + 1, y + 1),   # A (z-bond)
                (x, y + 1),       # B (y-bond from start)
            ]
        else:  # B site
            sites = [
                (x, y),
                (x + 1, y),
                (x + 1, y + 1),
                (x, y + 1),
                (x - 1, y + 1),
                (x - 1, y),
            ]
        
        return [self.wrap_position(s) for s in sites]
    
    def get_star_sites(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the sites in the star (vertex) operator region.
        
        For honeycomb, this is the 3 sites connected to the given site.
        
        Parameters
        ----------
        pos : tuple
            Center site position
        
        Returns
        -------
        list of tuple
            Positions of neighboring sites
        """
        directions = self.get_directions(pos)
        return [self.get_neighbor(pos, d) for d in directions if not d.startswith('-')]
    
    def __repr__(self) -> str:
        return (
            f"HoneycombLattice(spin_dim={self.spin_dim}, boson_dim={self.boson_dim}, "
            f"Lx={self.Lx//2}, Ly={self.Ly}, sites={self.num_sites}, "
            f"physical_dim={self._physical_dim})"
        )


class ExtendedHoneycombLattice(HoneycombLattice):
    """
    Extended honeycomb lattice with additional interactions.
    
    Supports:
    - Nearest-neighbor (NN) bonds
    - Next-nearest-neighbor (NNN) bonds
    - Third-nearest-neighbor bonds
    
    Useful for models with longer-range interactions like
    the J1-J2 model or Kitaev-Heisenberg model with extended terms.
    
    Parameters
    ----------
    include_nnn : bool
        Include next-nearest-neighbor bonds
    include_3nn : bool
        Include third-nearest-neighbor bonds
    """
    
    def __init__(
        self,
        spin_dim: int = 2,
        boson_dim: int = 1,
        Lx: int = 1,
        Ly: int = 1,
        include_nnn: bool = False,
        include_3nn: bool = False,
    ):
        self.include_nnn = include_nnn
        self.include_3nn = include_3nn
        super().__init__(spin_dim, boson_dim, Lx, Ly)
    
    def _initialize_bonds(self) -> None:
        """Initialize bonds including extended interactions."""
        # First add NN bonds
        super()._initialize_bonds()
        
        # Add NNN bonds if requested
        if self.include_nnn:
            self._add_nnn_bonds()
        
        # Add 3NN bonds if requested
        if self.include_3nn:
            self._add_3nn_bonds()
    
    def _add_nnn_bonds(self) -> None:
        """Add next-nearest-neighbor bonds."""
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                
                # NNN bonds within the same sublattice
                # For honeycomb, NNN connects sites of the same type
                
                if x % 2 == 0:  # A sublattice
                    # Connect to other A sites
                    nnn_neighbors = [
                        (x + 2, y),      # Along x
                        (x, y + 1),      # Along y (if same sublattice)
                        (x + 2, y + 1),  # Diagonal
                    ]
                else:  # B sublattice
                    nnn_neighbors = [
                        (x + 2, y),
                        (x, y + 1),
                        (x - 2, y + 1),
                    ]
                
                for neighbor in nnn_neighbors:
                    wrapped = self.wrap_position(neighbor)
                    if self.get_sublattice(pos) == self.get_sublattice(wrapped):
                        self._add_bond(pos, wrapped, 'nnn')
    
    def _add_3nn_bonds(self) -> None:
        """Add third-nearest-neighbor bonds."""
        for x in range(self.Lx):
            for y in range(self.Ly):
                pos = (x, y)
                
                # 3NN connects opposite sublattices at larger distance
                neighbors_3nn = [
                    (x + 3, y),
                    (x + 1, y + 1),
                    (x - 1, y + 1),
                ]
                
                for neighbor in neighbors_3nn:
                    wrapped = self.wrap_position(neighbor)
                    self._add_bond(pos, wrapped, '3nn')
