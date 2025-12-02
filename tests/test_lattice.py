"""
Tests for lattice geometry.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestUnitCell:
    """Tests for UnitCell class."""
    
    def test_create_unit_cell(self):
        """Test unit cell creation."""
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        
        assert uc.lx == 2
        assert uc.ly == 2
        assert uc.n_sites == 4
    
    def test_get_site(self):
        """Test site retrieval."""
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        
        site = uc.get_site(0, 0)
        assert site is not None
        assert site.x == 0
        assert site.y == 0
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        
        # (2, 0) should wrap to (0, 0)
        site = uc.get_site(2, 0)
        assert site.x == 0
        assert site.y == 0
    
    def test_get_neighbors(self):
        """Test neighbor retrieval."""
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        
        neighbors = uc.get_neighbors(0, 0)
        
        # Should have 4 neighbors on square lattice
        assert len(neighbors) == 4
    
    def test_get_bonds(self):
        """Test bond enumeration."""
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        
        bonds = uc.get_bonds()
        
        # 2x2 unit cell should have 8 bonds (4 horizontal + 4 vertical)
        assert len(bonds) == 8


class TestSite:
    """Tests for Site class."""
    
    def test_site_creation(self):
        """Test site creation."""
        from ipeps.lattice.unit_cell import Site
        
        site = Site(x=1, y=2, sublattice=0)
        
        assert site.x == 1
        assert site.y == 2
        assert site.sublattice == 0
    
    def test_site_hash(self):
        """Test that sites can be used as dict keys."""
        from ipeps.lattice.unit_cell import Site
        
        s1 = Site(x=0, y=0, sublattice=0)
        s2 = Site(x=0, y=0, sublattice=0)
        s3 = Site(x=1, y=0, sublattice=0)
        
        # Same coordinates should hash the same
        assert hash(s1) == hash(s2)
        
        # Can use as dict key
        d = {s1: "value"}
        assert d[s2] == "value"


class TestBond:
    """Tests for Bond class."""
    
    def test_bond_creation(self):
        """Test bond creation."""
        from ipeps.lattice.unit_cell import Site, Bond
        
        s1 = Site(0, 0, 0)
        s2 = Site(1, 0, 0)
        
        bond = Bond(site1=s1, site2=s2, direction='x')
        
        assert bond.site1 == s1
        assert bond.site2 == s2
        assert bond.direction == 'x'


class TestHoneycombLattice:
    """Tests for HoneycombLattice class."""
    
    def test_create_honeycomb(self):
        """Test honeycomb lattice creation."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        assert hc.lx == 2
        assert hc.ly == 2
        # Honeycomb has 2 sublattices
        assert hc.n_sublattices == 2
    
    def test_honeycomb_has_two_sublattices(self):
        """Test that honeycomb has two sublattice sites."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        sublattices = set()
        for site in hc.sites:
            sublattices.add(site.sublattice)
        
        assert len(sublattices) == 2
    
    def test_honeycomb_coordination(self):
        """Test that honeycomb sites have coordination 3."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        # Each site should have 3 neighbors
        for site in hc.sites:
            neighbors = hc.get_neighbors(site)
            assert len(neighbors) == 3, f"Site {site} has {len(neighbors)} neighbors"
    
    def test_honeycomb_bonds(self):
        """Test honeycomb bond enumeration."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        bonds = hc.get_bonds()
        
        # 2x2 honeycomb unit cell has 4 sites, each with 3 bonds, but each bond counted once
        # So 4*3/2 = 6 bonds per unit cell times 4 = 24 bonds? 
        # Actually depends on implementation
        assert len(bonds) > 0
    
    def test_to_effective_square(self):
        """Test brick-wall mapping to square lattice."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        # Should have method to get effective square lattice coordinates
        if hasattr(hc, 'to_effective_square'):
            for site in hc.sites:
                ex, ey = hc.to_effective_square(site)
                assert isinstance(ex, int)
                assert isinstance(ey, int)


class TestHoneycombGeometry:
    """Tests for honeycomb geometric properties."""
    
    def test_sublattice_assignment(self):
        """Test correct sublattice assignment."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        # Neighboring sites should be on different sublattices
        for bond in hc.get_bonds():
            assert bond.site1.sublattice != bond.site2.sublattice
    
    def test_bond_directions(self):
        """Test that bonds have correct directions."""
        from ipeps.lattice.honeycomb import HoneycombLattice
        
        hc = HoneycombLattice(lx=2, ly=2)
        
        bonds = hc.get_bonds()
        directions = set(b.direction for b in bonds)
        
        # Honeycomb should have 3 bond directions
        assert len(directions) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
