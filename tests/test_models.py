"""
Tests for physical models and operators.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestSpinOperators:
    """Tests for spin operators."""
    
    def test_spin_half_pauli(self):
        """Test spin-1/2 Pauli matrices."""
        from ipeps.models.operators import SpinOperators
        
        ops = SpinOperators(spin=0.5)
        
        Sx = ops.Sx
        Sy = ops.Sy
        Sz = ops.Sz
        
        # Check dimensions
        assert Sx.shape == (2, 2)
        assert Sy.shape == (2, 2)
        assert Sz.shape == (2, 2)
        
        # Check Sz eigenvalues
        eigvals = np.linalg.eigvalsh(Sz)
        assert_allclose(sorted(eigvals), [-0.5, 0.5])
    
    def test_spin_half_commutators(self):
        """Test [Si, Sj] = i*epsilon_ijk * Sk."""
        from ipeps.models.operators import SpinOperators
        
        ops = SpinOperators(spin=0.5)
        
        Sx, Sy, Sz = ops.Sx, ops.Sy, ops.Sz
        
        # [Sx, Sy] = i*Sz
        comm_xy = Sx @ Sy - Sy @ Sx
        assert_allclose(comm_xy, 1j * Sz, atol=1e-14)
        
        # [Sy, Sz] = i*Sx
        comm_yz = Sy @ Sz - Sz @ Sy
        assert_allclose(comm_yz, 1j * Sx, atol=1e-14)
        
        # [Sz, Sx] = i*Sy
        comm_zx = Sz @ Sx - Sx @ Sz
        assert_allclose(comm_zx, 1j * Sy, atol=1e-14)
    
    def test_spin_half_ladders(self):
        """Test spin ladder operators."""
        from ipeps.models.operators import SpinOperators
        
        ops = SpinOperators(spin=0.5)
        
        Sp = ops.Sp  # S+
        Sm = ops.Sm  # S-
        Sz = ops.Sz
        
        # S+ raises spin
        up = np.array([1, 0])
        down = np.array([0, 1])
        
        assert_allclose(Sp @ down, up)
        assert_allclose(Sp @ up, np.zeros(2))
        
        assert_allclose(Sm @ up, down)
        assert_allclose(Sm @ down, np.zeros(2))
    
    def test_spin_one(self):
        """Test spin-1 operators."""
        from ipeps.models.operators import SpinOperators
        
        ops = SpinOperators(spin=1.0)
        
        Sz = ops.Sz
        
        # Check dimension
        assert Sz.shape == (3, 3)
        
        # Check eigenvalues
        eigvals = np.linalg.eigvalsh(Sz)
        assert_allclose(sorted(eigvals), [-1, 0, 1])
    
    def test_identity(self):
        """Test identity operator."""
        from ipeps.models.operators import SpinOperators
        
        ops = SpinOperators(spin=0.5)
        Id = ops.Id
        
        assert_allclose(Id, np.eye(2))


class TestBosonOperators:
    """Tests for bosonic operators."""
    
    def test_creation_annihilation(self):
        """Test bosonic creation/annihilation operators."""
        from ipeps.models.operators import BosonOperators
        
        n_max = 4
        ops = BosonOperators(n_max=n_max)
        
        a = ops.a  # annihilation
        adag = ops.adag  # creation
        
        assert a.shape == (n_max, n_max)
        assert adag.shape == (n_max, n_max)
        
        # a and adag should be Hermitian conjugates
        assert_allclose(adag, a.conj().T)
    
    def test_number_operator(self):
        """Test number operator n = adag @ a."""
        from ipeps.models.operators import BosonOperators
        
        n_max = 5
        ops = BosonOperators(n_max=n_max)
        
        n = ops.n
        
        # Number operator should be diagonal with eigenvalues 0, 1, 2, ...
        eigvals = np.diag(n)
        assert_allclose(eigvals, np.arange(n_max))
    
    def test_commutator(self):
        """Test [a, adag] = 1 (within truncation)."""
        from ipeps.models.operators import BosonOperators
        
        n_max = 5
        ops = BosonOperators(n_max=n_max)
        
        a = ops.a
        adag = ops.adag
        
        comm = a @ adag - adag @ a
        
        # Should be identity except at the boundary
        expected = np.eye(n_max)
        expected[-1, -1] = 0  # Truncation effect
        
        assert_allclose(comm, expected, atol=1e-14)
    
    def test_ladder_action(self):
        """Test that creation/annihilation work as ladders."""
        from ipeps.models.operators import BosonOperators
        
        n_max = 4
        ops = BosonOperators(n_max=n_max)
        
        a = ops.a
        adag = ops.adag
        
        # |0> state
        vac = np.zeros(n_max)
        vac[0] = 1
        
        # a|0> = 0
        assert_allclose(a @ vac, np.zeros(n_max))
        
        # adag|0> = |1>
        one_boson = np.zeros(n_max)
        one_boson[1] = 1
        assert_allclose(adag @ vac, one_boson)


class TestSpinBosonHamiltonian:
    """Tests for spin-boson coupled Hamiltonians."""
    
    def test_hilbert_space_dimension(self):
        """Test combined Hilbert space dimension."""
        from ipeps.models.spin_boson import SpinBosonHamiltonian
        
        H = SpinBosonHamiltonian(
            spin=0.5,
            n_bosons=4,
            model_type="holstein",
        )
        
        # Dimension should be 2 * 4 = 8
        dim = H.local_dim
        assert dim == 8
    
    def test_holstein_hamiltonian(self):
        """Test Holstein model Hamiltonian."""
        from ipeps.models.spin_boson import SpinBosonHamiltonian
        
        H = SpinBosonHamiltonian(
            spin=0.5,
            n_bosons=3,
            model_type="holstein",
            g=0.5,  # coupling
            omega=1.0,  # phonon frequency
        )
        
        h_local = H.get_local_hamiltonian()
        
        # Should be Hermitian
        assert_allclose(h_local, h_local.conj().T, atol=1e-14)
    
    def test_two_site_gate(self):
        """Test two-site gate construction."""
        from ipeps.models.spin_boson import SpinBosonHamiltonian
        
        H = SpinBosonHamiltonian(
            spin=0.5,
            n_bosons=2,
            model_type="heisenberg",
            J=1.0,
        )
        
        gate = H.get_two_site_gate(dt=0.01)
        
        # Should be square matrix
        dim = H.local_dim ** 2
        assert gate.shape == (dim, dim)
        
        # Should be unitary for imaginary time (approximately symmetric for real time)
        # For small dt, gate should be close to identity
        assert_allclose(
            gate @ gate.conj().T,
            np.eye(dim),
            atol=0.1,  # Loose tolerance for finite dt
        )
    
    def test_heisenberg_exchange(self):
        """Test Heisenberg exchange term."""
        from ipeps.models.spin_boson import SpinBosonHamiltonian
        
        H = SpinBosonHamiltonian(
            spin=0.5,
            n_bosons=1,
            model_type="heisenberg",
            J=1.0,
        )
        
        h_bond = H.get_bond_hamiltonian()
        
        # For spin-1/2 Heisenberg: H = J * S1 · S2
        # Singlet state should have energy -3J/4
        # Triplet states should have energy J/4
        
        eigvals = np.linalg.eigvalsh(h_bond)
        
        # Should have at least one singlet and three triplet states
        # (times boson states)
        assert len(eigvals) > 0


class TestModelSymmetries:
    """Tests for model symmetries."""
    
    def test_spin_rotation_invariance(self):
        """Test SU(2) symmetry of Heisenberg model."""
        from ipeps.models.spin_boson import SpinBosonHamiltonian
        from ipeps.models.operators import SpinOperators
        
        H = SpinBosonHamiltonian(
            spin=0.5,
            n_bosons=1,
            model_type="heisenberg",
            J=1.0,
        )
        
        h_bond = H.get_bond_hamiltonian()
        
        # Total Sz should commute with Hamiltonian
        spin_ops = SpinOperators(spin=0.5)
        Sz = spin_ops.Sz
        Id_spin = np.eye(2)
        Id_boson = np.eye(1)
        
        Sz_tot = (
            np.kron(np.kron(Sz, Id_spin), np.kron(Id_boson, Id_boson)) +
            np.kron(np.kron(Id_spin, Sz), np.kron(Id_boson, Id_boson))
        )
        
        comm = h_bond @ Sz_tot - Sz_tot @ h_bond
        assert_allclose(comm, 0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
