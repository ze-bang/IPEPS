"""
Tests for core tensor operations.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestTensor:
    """Tests for the Tensor class."""
    
    def test_creation_from_ndarray(self):
        """Test tensor creation from NumPy array."""
        from ipeps.core.tensor import Tensor
        
        data = np.random.randn(3, 4, 5)
        t = Tensor(data)
        
        assert t.shape == (3, 4, 5)
        assert t.ndim == 3
        assert t.size == 60
        assert_array_equal(t.data, data)
    
    def test_creation_random(self):
        """Test random tensor creation."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor.random((4, 5, 6), dtype=np.complex128)
        
        assert t.shape == (4, 5, 6)
        assert t.dtype == np.complex128
        # Check it's actually random (not all zeros)
        assert np.abs(t.data).sum() > 0
    
    def test_creation_zeros(self):
        """Test zero tensor creation."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor.zeros((2, 3, 4))
        
        assert t.shape == (2, 3, 4)
        assert_array_equal(t.data, np.zeros((2, 3, 4)))
    
    def test_creation_ones(self):
        """Test ones tensor creation."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor.ones((2, 3))
        
        assert t.shape == (2, 3)
        assert_array_equal(t.data, np.ones((2, 3)))
    
    def test_reshape(self):
        """Test tensor reshape."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor.random((2, 3, 4))
        t_reshaped = t.reshape((6, 4))
        
        assert t_reshaped.shape == (6, 4)
        assert t_reshaped.size == t.size
    
    def test_transpose(self):
        """Test tensor transpose."""
        from ipeps.core.tensor import Tensor
        
        data = np.random.randn(2, 3, 4)
        t = Tensor(data)
        t_transposed = t.transpose((2, 0, 1))
        
        assert t_transposed.shape == (4, 2, 3)
        assert_array_equal(t_transposed.data, data.transpose((2, 0, 1)))
    
    def test_conjugate(self):
        """Test complex conjugation."""
        from ipeps.core.tensor import Tensor
        
        data = np.array([1+2j, 3-4j, 5+6j])
        t = Tensor(data)
        t_conj = t.conj()
        
        assert_array_equal(t_conj.data, np.array([1-2j, 3+4j, 5-6j]))
    
    def test_norm(self):
        """Test Frobenius norm."""
        from ipeps.core.tensor import Tensor
        
        data = np.array([3.0, 4.0])
        t = Tensor(data)
        
        assert_allclose(t.norm(), 5.0)
    
    def test_normalize(self):
        """Test tensor normalization."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor(np.array([3.0, 4.0]))
        t_normalized = t.normalize()
        
        assert_allclose(t_normalized.norm(), 1.0)
    
    def test_contract_matrix_multiply(self):
        """Test tensor contraction (matrix multiplication case)."""
        from ipeps.core.tensor import Tensor
        
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(4, 5))
        
        c = a.contract(b, ([1], [0]))
        
        assert c.shape == (3, 5)
        assert_allclose(c.data, a.data @ b.data)
    
    def test_contract_tensordot(self):
        """Test general tensor contraction."""
        from ipeps.core.tensor import Tensor
        
        a = Tensor(np.random.randn(2, 3, 4))
        b = Tensor(np.random.randn(4, 5, 3))
        
        c = a.contract(b, ([1, 2], [2, 0]))
        
        assert c.shape == (2, 5)
        expected = np.tensordot(a.data, b.data, axes=([1, 2], [2, 0]))
        assert_allclose(c.data, expected)
    
    def test_svd(self):
        """Test SVD decomposition."""
        from ipeps.core.tensor import Tensor
        
        data = np.random.randn(6, 8)
        t = Tensor(data)
        
        U, S, Vh = t.svd()
        
        # Check shapes
        assert U.shape[0] == 6
        assert Vh.shape[1] == 8
        
        # Check reconstruction
        reconstructed = U.data @ np.diag(S.data) @ Vh.data
        assert_allclose(reconstructed, data)
    
    def test_svd_truncated(self):
        """Test truncated SVD."""
        from ipeps.core.tensor import Tensor
        
        data = np.random.randn(10, 8)
        t = Tensor(data)
        
        U, S, Vh = t.svd(max_bond=4)
        
        assert S.shape == (4,)
        assert U.shape == (10, 4)
        assert Vh.shape == (4, 8)
    
    def test_qr(self):
        """Test QR decomposition."""
        from ipeps.core.tensor import Tensor
        
        data = np.random.randn(5, 4)
        t = Tensor(data)
        
        Q, R = t.qr()
        
        # Check orthogonality of Q
        assert_allclose(Q.data.T.conj() @ Q.data, np.eye(4), atol=1e-10)
        
        # Check reconstruction
        assert_allclose(Q.data @ R.data, data)
    
    def test_add(self):
        """Test tensor addition."""
        from ipeps.core.tensor import Tensor
        
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0, 6.0]))
        
        c = a + b
        
        assert_array_equal(c.data, np.array([5.0, 7.0, 9.0]))
    
    def test_multiply(self):
        """Test tensor multiplication by scalar."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        scaled = t * 2
        
        assert_array_equal(scaled.data, np.array([2.0, 4.0, 6.0]))
    
    def test_copy(self):
        """Test tensor copy."""
        from ipeps.core.tensor import Tensor
        
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t_copy = t.copy()
        
        # Modify original
        t.data[0] = 99.0
        
        # Copy should be unchanged
        assert t_copy.data[0] == 1.0


class TestBackend:
    """Tests for backend switching."""
    
    def test_numpy_backend(self):
        """Test NumPy backend."""
        from ipeps.core.tensor import set_backend, get_backend
        
        set_backend('numpy')
        assert get_backend() == 'numpy'
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        from ipeps.core.tensor import set_backend
        
        with pytest.raises(ValueError):
            set_backend('invalid_backend')


class TestContractions:
    """Tests for contraction operations."""
    
    def test_contract_function(self):
        """Test contract function."""
        from ipeps.core.contractions import contract
        
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        
        c = contract("ij,jk->ik", a, b)
        
        assert c.shape == (3, 5)
        assert_allclose(c, a @ b)
    
    def test_trace(self):
        """Test trace operation."""
        from ipeps.core.contractions import trace_indices
        
        # Create tensor with equal first and last dimensions
        a = np.random.randn(3, 4, 3)
        
        traced = trace_indices(a, [0, 2])
        
        assert traced.shape == (4,)
        # Manual trace
        expected = np.einsum("iji->j", a)
        assert_allclose(traced, expected)


class TestDecompositions:
    """Tests for tensor decompositions."""
    
    def test_truncated_svd(self):
        """Test truncated SVD."""
        from ipeps.core.decompositions import truncated_svd
        
        m = np.random.randn(10, 8)
        U, S, Vh = truncated_svd(m, max_bond=4)
        
        assert S.shape == (4,)
        assert U.shape == (10, 4)
        assert Vh.shape == (4, 8)
    
    def test_eigh_decomposition(self):
        """Test Hermitian eigendecomposition."""
        from ipeps.core.decompositions import eigh_truncated
        
        # Create Hermitian matrix
        m = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
        m = (m + m.conj().T) / 2
        
        w, v = eigh_truncated(m, n_eigs=3)
        
        assert w.shape == (3,)
        assert v.shape == (5, 3)
        
        # Check eigenvalue equation for largest eigenvalue
        idx = 0  # First column has largest eigenvalue
        assert_allclose(
            m @ v[:, idx],
            w[idx] * v[:, idx],
            atol=1e-10,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
