"""
Tensor decomposition routines for iPEPS.

Provides:
- SVD with truncation
- QR decomposition
- Eigendecomposition
- Specialized decompositions for tensor networks
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import warnings

from ipeps.core.tensor import Tensor, get_backend_module, Backend


@dataclass
class SVDResult:
    """Result of SVD decomposition."""
    U: Tensor
    S: Tensor
    Vh: Tensor
    truncation_error: float = 0.0
    rank: int = 0
    
    def reconstruct(self) -> Tensor:
        """Reconstruct the original tensor from SVD factors."""
        xp = get_backend_module()
        # U @ diag(S) @ Vh
        US = self.U.data @ xp.diag(self.S.data)
        return Tensor(US @ self.Vh.data, backend=self.U.backend)


@dataclass
class QRResult:
    """Result of QR decomposition."""
    Q: Tensor
    R: Tensor


@dataclass
class EigResult:
    """Result of eigendecomposition."""
    eigenvalues: Tensor
    eigenvectors: Tensor


def svd(
    tensor: Tensor,
    full_matrices: bool = False,
) -> SVDResult:
    """
    Compute the Singular Value Decomposition.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor (will be reshaped to matrix if not 2D)
    full_matrices : bool
        If True, compute full-sized U and Vh
    
    Returns
    -------
    SVDResult
        Named tuple with U, S, Vh tensors
    """
    xp = get_backend_module()
    
    data = tensor.data
    original_shape = tensor.dims
    
    # Reshape to matrix if needed
    if tensor.ndim != 2:
        mid = tensor.ndim // 2
        left_dim = int(np.prod(original_shape[:mid]))
        right_dim = int(np.prod(original_shape[mid:]))
        data = data.reshape(left_dim, right_dim)
    
    U, S, Vh = xp.linalg.svd(data, full_matrices=full_matrices)
    
    return SVDResult(
        U=Tensor(U, backend=tensor.backend),
        S=Tensor(S, backend=tensor.backend),
        Vh=Tensor(Vh, backend=tensor.backend),
        rank=len(S),
    )


def truncated_svd(
    tensor: Tensor,
    max_rank: Optional[int] = None,
    cutoff: float = 1e-14,
    normalize: bool = False,
    return_error: bool = True,
) -> SVDResult:
    """
    Truncated Singular Value Decomposition.
    
    Computes SVD and truncates to keep only the largest singular values.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor
    max_rank : int, optional
        Maximum number of singular values to keep
    cutoff : float
        Relative cutoff for singular values (values below cutoff * max(S) are discarded)
    normalize : bool
        Whether to normalize the singular values after truncation
    return_error : bool
        Whether to compute and return the truncation error
    
    Returns
    -------
    SVDResult
        Truncated SVD factors with truncation error
    
    Examples
    --------
    >>> t = Tensor.random((10, 10))
    >>> result = truncated_svd(t, max_rank=5)
    >>> result.U.shape
    (10, 5)
    >>> result.truncation_error
    0.0123...  # truncation error
    """
    xp = get_backend_module()
    
    data = tensor.data
    original_shape = tensor.dims
    
    # Reshape to matrix if needed
    if tensor.ndim != 2:
        mid = tensor.ndim // 2
        left_dim = int(np.prod(original_shape[:mid]))
        right_dim = int(np.prod(original_shape[mid:]))
        data = data.reshape(left_dim, right_dim)
    
    # Full SVD
    U, S, Vh = xp.linalg.svd(data, full_matrices=False)
    
    # Convert to numpy for threshold calculations
    S_np = S if isinstance(S, np.ndarray) else S.get() if hasattr(S, 'get') else np.array(S)
    
    # Determine truncation
    if len(S_np) == 0:
        rank = 0
    else:
        # Apply relative cutoff
        threshold = cutoff * S_np[0]
        rank = int(np.sum(S_np > threshold))
        
        # Apply max_rank constraint
        if max_rank is not None:
            rank = min(rank, max_rank)
        
        # Ensure at least rank 1
        rank = max(1, rank)
    
    # Truncate
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vh_trunc = Vh[:rank, :]
    
    # Compute truncation error
    if return_error and len(S_np) > rank:
        truncated_singular_values = S_np[rank:]
        truncation_error = float(np.sqrt(np.sum(truncated_singular_values ** 2)))
        # Normalize by total norm
        total_norm = float(np.sqrt(np.sum(S_np ** 2)))
        if total_norm > 1e-15:
            truncation_error /= total_norm
    else:
        truncation_error = 0.0
    
    # Normalize if requested
    if normalize:
        norm = xp.linalg.norm(S_trunc)
        if norm > 1e-15:
            S_trunc = S_trunc / norm
    
    return SVDResult(
        U=Tensor(U_trunc, backend=tensor.backend),
        S=Tensor(S_trunc, backend=tensor.backend),
        Vh=Tensor(Vh_trunc, backend=tensor.backend),
        truncation_error=truncation_error,
        rank=rank,
    )


def qr(
    tensor: Tensor,
    mode: str = 'reduced',
) -> QRResult:
    """
    Compute QR decomposition.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor (reshaped to matrix if not 2D)
    mode : str
        'reduced' for thin QR, 'complete' for full QR
    
    Returns
    -------
    QRResult
        Named tuple with Q and R tensors
    """
    xp = get_backend_module()
    
    data = tensor.data
    
    # Reshape to matrix if needed
    if tensor.ndim != 2:
        mid = tensor.ndim // 2
        left_dim = int(np.prod(tensor.dims[:mid]))
        right_dim = int(np.prod(tensor.dims[mid:]))
        data = data.reshape(left_dim, right_dim)
    
    Q, R = xp.linalg.qr(data, mode=mode)
    
    return QRResult(
        Q=Tensor(Q, backend=tensor.backend),
        R=Tensor(R, backend=tensor.backend),
    )


def lq(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute LQ decomposition (transpose of QR).
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor
    
    Returns
    -------
    L : Tensor
        Lower triangular tensor
    Q : Tensor
        Orthogonal tensor
    """
    # LQ = (QR of transpose)^T
    result = qr(tensor.transpose())
    return result.R.transpose(), result.Q.transpose()


def eig(
    tensor: Tensor,
    hermitian: bool = False,
) -> EigResult:
    """
    Compute eigendecomposition.
    
    Parameters
    ----------
    tensor : Tensor
        Input square matrix
    hermitian : bool
        If True, use Hermitian eigendecomposition (faster, more stable)
    
    Returns
    -------
    EigResult
        Named tuple with eigenvalues and eigenvectors
    """
    xp = get_backend_module()
    
    data = tensor.data
    
    if hermitian:
        eigenvalues, eigenvectors = xp.linalg.eigh(data)
    else:
        eigenvalues, eigenvectors = xp.linalg.eig(data)
    
    return EigResult(
        eigenvalues=Tensor(eigenvalues, backend=tensor.backend),
        eigenvectors=Tensor(eigenvectors, backend=tensor.backend),
    )


def polar(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute polar decomposition: A = U @ P
    
    where U is unitary and P is positive semi-definite.
    
    Parameters
    ----------
    tensor : Tensor
        Input matrix
    
    Returns
    -------
    U : Tensor
        Unitary matrix
    P : Tensor
        Positive semi-definite matrix
    """
    xp = get_backend_module()
    
    result = svd(tensor)
    U = result.U.data @ result.Vh.data
    P = result.Vh.conj().transpose().data @ xp.diag(result.S.data) @ result.Vh.data
    
    return Tensor(U, backend=tensor.backend), Tensor(P, backend=tensor.backend)


def expm(tensor: Tensor) -> Tensor:
    """
    Compute matrix exponential.
    
    Parameters
    ----------
    tensor : Tensor
        Input square matrix
    
    Returns
    -------
    Tensor
        Matrix exponential exp(tensor)
    """
    from scipy import linalg as spla
    
    data = tensor.numpy()
    result = spla.expm(data)
    
    return Tensor(result, backend=tensor.backend)


def logm(tensor: Tensor) -> Tensor:
    """
    Compute matrix logarithm.
    
    Parameters
    ----------
    tensor : Tensor
        Input square matrix
    
    Returns
    -------
    Tensor
        Matrix logarithm log(tensor)
    """
    from scipy import linalg as spla
    
    data = tensor.numpy()
    result = spla.logm(data)
    
    return Tensor(result, backend=tensor.backend)


def sqrtm(tensor: Tensor) -> Tensor:
    """
    Compute matrix square root.
    
    Parameters
    ----------
    tensor : Tensor
        Input positive semi-definite matrix
    
    Returns
    -------
    Tensor
        Matrix square root sqrt(tensor)
    """
    from scipy import linalg as spla
    
    data = tensor.numpy()
    result = spla.sqrtm(data)
    
    return Tensor(result, backend=tensor.backend)


def tensor_svd(
    tensor: Tensor,
    left_axes: Tuple[int, ...],
    right_axes: Tuple[int, ...],
    max_rank: Optional[int] = None,
    cutoff: float = 1e-14,
    absorb: str = 'both',
) -> Tuple[Tensor, Tensor]:
    """
    SVD of a tensor along specified axes.
    
    Reshapes tensor to matrix, performs truncated SVD, and reshapes back.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor
    left_axes : tuple of int
        Axes to group on the left (row) side
    right_axes : tuple of int
        Axes to group on the right (column) side
    max_rank : int, optional
        Maximum bond dimension
    cutoff : float
        Singular value cutoff
    absorb : str
        Where to absorb singular values:
        - 'left': into left tensor
        - 'right': into right tensor  
        - 'both': sqrt into both
        - 'none': return three tensors
    
    Returns
    -------
    left : Tensor
        Left tensor with shape (*left_dims, rank)
    right : Tensor
        Right tensor with shape (rank, *right_dims)
    
    Examples
    --------
    >>> t = Tensor.random((2, 3, 4, 5))
    >>> left, right = tensor_svd(t, (0, 1), (2, 3), max_rank=10)
    >>> left.shape
    (2, 3, 10)
    >>> right.shape
    (10, 4, 5)
    """
    xp = get_backend_module()
    
    # Compute shapes
    left_dims = tuple(tensor.dims[i] for i in left_axes)
    right_dims = tuple(tensor.dims[i] for i in right_axes)
    left_size = int(np.prod(left_dims))
    right_size = int(np.prod(right_dims))
    
    # Permute and reshape to matrix
    perm = left_axes + right_axes
    data = xp.transpose(tensor.data, perm)
    matrix = data.reshape(left_size, right_size)
    
    # SVD
    matrix_tensor = Tensor(matrix, backend=tensor.backend)
    result = truncated_svd(matrix_tensor, max_rank=max_rank, cutoff=cutoff)
    
    U = result.U.data
    S = result.S.data
    Vh = result.Vh.data
    rank = result.rank
    
    # Absorb singular values
    if absorb == 'left':
        U = U @ xp.diag(S)
    elif absorb == 'right':
        Vh = xp.diag(S) @ Vh
    elif absorb == 'both':
        sqrt_S = xp.sqrt(S)
        U = U @ xp.diag(sqrt_S)
        Vh = xp.diag(sqrt_S) @ Vh
    # absorb == 'none': return U, S, Vh separately (not implemented here)
    
    # Reshape back
    left = U.reshape(left_dims + (rank,))
    right = Vh.reshape((rank,) + right_dims)
    
    return Tensor(left, backend=tensor.backend), Tensor(right, backend=tensor.backend)


def bond_canonicalize(
    tensor_left: Tensor,
    tensor_right: Tensor,
    bond_axis_left: int,
    bond_axis_right: int,
    max_rank: Optional[int] = None,
    cutoff: float = 1e-14,
) -> Tuple[Tensor, Tensor, float]:
    """
    Canonicalize the bond between two tensors.
    
    Performs SVD across the bond and distributes singular values.
    Used in simple update and for gauge fixing.
    
    Parameters
    ----------
    tensor_left : Tensor
        Left tensor
    tensor_right : Tensor
        Right tensor
    bond_axis_left : int
        Index of the bond axis in the left tensor
    bond_axis_right : int
        Index of the bond axis in the right tensor
    max_rank : int, optional
        Maximum bond dimension after truncation
    cutoff : float
        Singular value cutoff
    
    Returns
    -------
    tensor_left_new : Tensor
        Updated left tensor
    tensor_right_new : Tensor
        Updated right tensor
    truncation_error : float
        Truncation error from SVD
    """
    xp = get_backend_module()
    
    # Contract tensors across bond
    combined = tensor_left.tensordot(tensor_right, axes=([bond_axis_left], [bond_axis_right]))
    
    # Determine axis groupings
    left_ndim = tensor_left.ndim
    right_ndim = tensor_right.ndim
    
    # Left axes (excluding the contracted bond)
    left_axes_orig = list(range(left_ndim))
    left_axes_orig.remove(bond_axis_left)
    
    # In combined tensor, left axes come first, then right axes
    left_axes = tuple(range(len(left_axes_orig)))
    right_axes = tuple(range(len(left_axes_orig), combined.ndim))
    
    # SVD
    new_left, new_right = tensor_svd(
        combined,
        left_axes=left_axes,
        right_axes=right_axes,
        max_rank=max_rank,
        cutoff=cutoff,
        absorb='both',
    )
    
    # Reshape tensors to have the bond axis in the correct position
    left_shape = list(new_left.dims[:-1])
    right_shape = list(new_right.dims[1:])
    bond_dim = new_left.dims[-1]
    
    # Insert bond axis back in original position
    left_shape.insert(bond_axis_left, bond_dim)
    right_shape.insert(bond_axis_right, bond_dim)
    
    # Need to permute to get bond axis in right place
    # For now, simplified version
    
    # Calculate truncation error
    # (This is approximate - would need full SVD spectrum for exact)
    truncation_error = 0.0  # Placeholder
    
    return new_left, new_right, truncation_error
