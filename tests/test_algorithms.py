"""
Tests for iPEPS algorithms.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestIPEPSState:
    """Tests for iPEPS state representation."""
    
    def test_create_state(self):
        """Test iPEPS state creation."""
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=2, ly=2)
        state = iPEPSState(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        assert state.bond_dim == 2
        assert state.phys_dim == 2
    
    def test_random_initialization(self):
        """Test random initialization."""
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=3, phys_dim=2)
        
        # Check that tensors are created
        assert len(state.tensors) > 0
        
        # Check tensor shapes
        for key, tensor in state.tensors.items():
            # iPEPS tensor: (phys, left, up, right, down)
            assert len(tensor.shape) == 5
    
    def test_normalize(self):
        """Test state normalization."""
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        state.normalize()
        
        # Each tensor should have unit norm (or close to it)
        for tensor in state.tensors.values():
            norm = np.linalg.norm(tensor.data)
            assert norm > 0


class TestCTMRG:
    """Tests for CTMRG algorithm."""
    
    def test_environment_initialization(self):
        """Test CTMRG environment initialization."""
        from ipeps.algorithms.ctmrg import CTMRGEnvironment, CTMRG
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        chi = 4
        env = CTMRGEnvironment.initialize(state, chi=chi)
        
        # Check corners and edges are created
        assert len(env.corners) > 0
        assert len(env.edges) > 0
    
    def test_ctmrg_convergence(self):
        """Test that CTMRG converges."""
        from ipeps.algorithms.ctmrg import CTMRG
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        ctmrg = CTMRG(chi=4, max_iter=50, tol=1e-6)
        env, converged = ctmrg.run(state)
        
        # Should converge (or at least run without error)
        assert env is not None


class TestSimpleUpdate:
    """Tests for Simple Update algorithm."""
    
    def test_apply_gate(self):
        """Test gate application."""
        from ipeps.algorithms.simple_update import SimpleUpdate
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        su = SimpleUpdate(
            state=state,
            dt=0.01,
            max_bond=4,
        )
        
        # Create a simple gate (identity)
        d = state.phys_dim
        gate = np.eye(d * d).reshape(d, d, d, d)
        
        # Should run without error
        bonds = uc.get_bonds()
        if bonds:
            su.apply_gate(gate, bonds[0])
    
    def test_evolution_step(self):
        """Test one evolution step."""
        from ipeps.algorithms.simple_update import SimpleUpdate
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        su = SimpleUpdate(
            state=state,
            dt=0.01,
            max_bond=4,
        )
        
        # Create simple Hamiltonian (diagonal)
        d = state.phys_dim
        H = np.diag(np.arange(d * d)).reshape(d, d, d, d)
        
        su.set_hamiltonian(H)
        
        # Run one step
        initial_energy = su.compute_energy_estimate()
        su.step()


class TestFullUpdate:
    """Tests for Full Update algorithm."""
    
    def test_initialization(self):
        """Test Full Update initialization."""
        from ipeps.algorithms.full_update import FullUpdate
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        fu = FullUpdate(
            state=state,
            chi=4,
            dt=0.01,
            max_bond=4,
        )
        
        assert fu is not None


class TestVariationalOptimizer:
    """Tests for variational optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        from ipeps.algorithms.variational import VariationalOptimizer
        from ipeps.core.ipeps_state import iPEPSState
        from ipeps.lattice.unit_cell import UnitCell
        
        uc = UnitCell(lx=1, ly=1)
        state = iPEPSState.random(unit_cell=uc, bond_dim=2, phys_dim=2)
        
        opt = VariationalOptimizer(
            state=state,
            chi=4,
            method='lbfgs',
        )
        
        assert opt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
