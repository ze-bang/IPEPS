"""
Core Tensor class with backend abstraction for HPC operations.

Supports:
- NumPy backend (default)
- CuPy backend (GPU acceleration)
- PyTorch backend (automatic differentiation)

The Tensor class provides a unified interface for tensor operations
that can be transparently executed on different hardware backends.
"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional, Tuple, List, Sequence, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import warnings

if TYPE_CHECKING:
    import cupy as cp
    import torch


class Backend(Enum):
    """Available computational backends."""
    NUMPY = "numpy"
    CUPY = "cupy"
    TORCH = "torch"


# Global backend state
_CURRENT_BACKEND: Backend = Backend.NUMPY
_BACKEND_MODULES: dict = {}


def get_backend() -> Backend:
    """Get the current computational backend."""
    return _CURRENT_BACKEND


def set_backend(backend: Union[Backend, str]) -> None:
    """
    Set the computational backend.
    
    Parameters
    ----------
    backend : Backend or str
        The backend to use ("numpy", "cupy", or "torch")
    """
    global _CURRENT_BACKEND
    
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    
    if backend == Backend.CUPY:
        try:
            import cupy as cp
            _BACKEND_MODULES['cupy'] = cp
        except ImportError:
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")
    elif backend == Backend.TORCH:
        try:
            import torch
            _BACKEND_MODULES['torch'] = torch
        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")
    
    _CURRENT_BACKEND = backend


def get_backend_module():
    """Get the current backend module (numpy, cupy, or torch)."""
    if _CURRENT_BACKEND == Backend.NUMPY:
        return np
    elif _CURRENT_BACKEND == Backend.CUPY:
        return _BACKEND_MODULES.get('cupy', np)
    elif _CURRENT_BACKEND == Backend.TORCH:
        return _BACKEND_MODULES.get('torch', np)
    return np


@dataclass
class TensorShape:
    """Represents the shape of a tensor with named dimensions."""
    dims: Tuple[int, ...]
    names: Optional[Tuple[str, ...]] = None
    
    def __post_init__(self):
        if self.names is not None and len(self.names) != len(self.dims):
            raise ValueError("Number of dimension names must match number of dimensions")
    
    @property
    def ndim(self) -> int:
        return len(self.dims)
    
    @property
    def total_size(self) -> int:
        return int(np.prod(self.dims))
    
    def __getitem__(self, idx: int) -> int:
        return self.dims[idx]
    
    def __len__(self) -> int:
        return len(self.dims)
    
    def __iter__(self):
        return iter(self.dims)


class Tensor:
    """
    Core tensor class with backend abstraction.
    
    This class wraps tensor data and provides a unified interface for
    tensor operations across different computational backends (NumPy, CuPy, PyTorch).
    
    Parameters
    ----------
    data : array_like
        The tensor data
    dtype : dtype, optional
        Data type (default: complex128 for quantum tensors)
    backend : Backend, optional
        Computational backend (default: current global backend)
    requires_grad : bool, optional
        Whether to track gradients (only for torch backend)
    labels : tuple of str, optional
        Names for each dimension (useful for ncon-style contractions)
    
    Attributes
    ----------
    data : ndarray
        The underlying tensor data
    shape : TensorShape
        Shape information including optional dimension names
    dtype : dtype
        Data type of the tensor
    backend : Backend
        The computational backend being used
    
    Examples
    --------
    >>> t = Tensor(np.random.randn(2, 3, 4))
    >>> t.shape
    TensorShape(dims=(2, 3, 4))
    >>> t.norm()
    5.123...  # Example norm value
    
    >>> # Named dimensions for clearer contractions
    >>> t = Tensor(data, labels=('left', 'physical', 'right'))
    """
    
    __slots__ = ('_data', '_shape', '_backend', '_labels', '_requires_grad')
    
    def __init__(
        self,
        data: Any,
        dtype: Optional[Any] = None,
        backend: Optional[Backend] = None,
        requires_grad: bool = False,
        labels: Optional[Tuple[str, ...]] = None,
    ):
        self._backend = backend or get_backend()
        self._requires_grad = requires_grad
        self._labels = labels
        
        # Default to complex128 for quantum simulations
        if dtype is None:
            dtype = np.complex128
        
        # Convert data to appropriate backend format
        self._data = self._to_backend(data, dtype)
        self._shape = TensorShape(tuple(self._data.shape), labels)
    
    def _to_backend(self, data: Any, dtype: Any) -> Any:
        """Convert data to the appropriate backend format."""
        if self._backend == Backend.NUMPY:
            if hasattr(data, 'get'):  # CuPy array
                data = data.get()
            elif hasattr(data, 'cpu'):  # PyTorch tensor
                data = data.cpu().numpy()
            return np.asarray(data, dtype=dtype)
        
        elif self._backend == Backend.CUPY:
            cp = _BACKEND_MODULES['cupy']
            if hasattr(data, 'cpu'):  # PyTorch tensor
                data = data.cpu().numpy()
            return cp.asarray(data, dtype=dtype)
        
        elif self._backend == Backend.TORCH:
            torch = _BACKEND_MODULES['torch']
            if hasattr(data, 'get'):  # CuPy array
                data = data.get()
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = torch.tensor(data, dtype=self._numpy_to_torch_dtype(dtype))
            if self._requires_grad:
                tensor.requires_grad_(True)
            return tensor
        
        return np.asarray(data, dtype=dtype)
    
    @staticmethod
    def _numpy_to_torch_dtype(dtype):
        """Convert numpy dtype to torch dtype."""
        torch = _BACKEND_MODULES.get('torch')
        if torch is None:
            return dtype
        
        dtype_map = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            np.int32: torch.int32,
            np.int64: torch.int64,
        }
        return dtype_map.get(dtype, torch.complex128)
    
    @property
    def data(self) -> Any:
        """Get the underlying tensor data."""
        return self._data
    
    @property
    def shape(self) -> TensorShape:
        """Get the tensor shape."""
        return self._shape
    
    @property
    def dims(self) -> Tuple[int, ...]:
        """Get the tensor dimensions as a tuple."""
        return self._shape.dims
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._shape.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._shape.total_size
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self._data.dtype
    
    @property
    def backend(self) -> Backend:
        """Computational backend."""
        return self._backend
    
    @property
    def labels(self) -> Optional[Tuple[str, ...]]:
        """Dimension labels."""
        return self._labels
    
    @property
    def T(self) -> 'Tensor':
        """Transpose of the tensor."""
        return self.transpose()
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.dims}, dtype={self.dtype}, backend={self._backend.value})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    # ==================== Factory Methods ====================
    
    @classmethod
    def zeros(
        cls,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
        backend: Optional[Backend] = None,
        labels: Optional[Tuple[str, ...]] = None,
    ) -> 'Tensor':
        """Create a tensor of zeros."""
        xp = get_backend_module()
        data = xp.zeros(shape, dtype=dtype)
        return cls(data, dtype=dtype, backend=backend, labels=labels)
    
    @classmethod
    def ones(
        cls,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
        backend: Optional[Backend] = None,
        labels: Optional[Tuple[str, ...]] = None,
    ) -> 'Tensor':
        """Create a tensor of ones."""
        xp = get_backend_module()
        data = xp.ones(shape, dtype=dtype)
        return cls(data, dtype=dtype, backend=backend, labels=labels)
    
    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
        backend: Optional[Backend] = None,
        labels: Optional[Tuple[str, ...]] = None,
        seed: Optional[int] = None,
    ) -> 'Tensor':
        """Create a random tensor."""
        if seed is not None:
            np.random.seed(seed)
        
        xp = get_backend_module()
        if np.issubdtype(dtype, np.complexfloating):
            real = xp.random.randn(*shape)
            imag = xp.random.randn(*shape)
            data = (real + 1j * imag) / np.sqrt(2)
        else:
            data = xp.random.randn(*shape)
        
        return cls(data, dtype=dtype, backend=backend, labels=labels)
    
    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Any = np.complex128,
        backend: Optional[Backend] = None,
    ) -> 'Tensor':
        """Create an identity matrix."""
        xp = get_backend_module()
        data = xp.eye(dim, dtype=dtype)
        return cls(data, dtype=dtype, backend=backend)
    
    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        backend: Optional[Backend] = None,
        labels: Optional[Tuple[str, ...]] = None,
    ) -> 'Tensor':
        """Create a tensor from a NumPy array."""
        return cls(array, dtype=array.dtype, backend=backend, labels=labels)
    
    # ==================== Conversion Methods ====================
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        if self._backend == Backend.NUMPY:
            return self._data
        elif self._backend == Backend.CUPY:
            return self._data.get()
        elif self._backend == Backend.TORCH:
            return self._data.detach().cpu().numpy()
        return np.asarray(self._data)
    
    def to(self, backend: Backend) -> 'Tensor':
        """Convert tensor to a different backend."""
        if backend == self._backend:
            return self
        return Tensor(self.numpy(), dtype=self.dtype, backend=backend, labels=self._labels)
    
    def clone(self) -> 'Tensor':
        """Create a deep copy of the tensor."""
        xp = get_backend_module()
        data_copy = xp.copy(self._data)
        return Tensor(data_copy, dtype=self.dtype, backend=self._backend, labels=self._labels)
    
    def copy(self) -> 'Tensor':
        """Alias for clone()."""
        return self.clone()
    
    # ==================== Basic Operations ====================
    
    def __add__(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data, backend=self._backend)
        return Tensor(self._data + other, backend=self._backend)
    
    def __radd__(self, other: Union[float, complex]) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data, backend=self._backend)
        return Tensor(self._data - other, backend=self._backend)
    
    def __rsub__(self, other: Union[float, complex]) -> 'Tensor':
        return Tensor(other - self._data, backend=self._backend)
    
    def __mul__(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data, backend=self._backend)
        return Tensor(self._data * other, backend=self._backend)
    
    def __rmul__(self, other: Union[float, complex]) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data, backend=self._backend)
        return Tensor(self._data / other, backend=self._backend)
    
    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data, backend=self._backend)
    
    def __abs__(self) -> 'Tensor':
        xp = get_backend_module()
        return Tensor(xp.abs(self._data), backend=self._backend)
    
    def __getitem__(self, idx) -> 'Tensor':
        return Tensor(self._data[idx], backend=self._backend)
    
    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self._data[idx] = value._data
        else:
            self._data[idx] = value
    
    # ==================== Linear Algebra Operations ====================
    
    def norm(self, ord: Optional[Union[int, float, str]] = None) -> float:
        """Compute the norm of the tensor."""
        xp = get_backend_module()
        if self._backend == Backend.TORCH:
            return float(self._data.norm(ord).item())
        return float(xp.linalg.norm(self._data, ord=ord))
    
    def normalize(self, inplace: bool = False) -> 'Tensor':
        """Normalize the tensor to have unit norm."""
        n = self.norm()
        if n < 1e-15:
            warnings.warn("Normalizing a near-zero tensor")
            return self if inplace else self.clone()
        
        if inplace:
            self._data /= n
            return self
        return self / n
    
    def conj(self) -> 'Tensor':
        """Complex conjugate."""
        xp = get_backend_module()
        return Tensor(xp.conj(self._data), backend=self._backend, labels=self._labels)
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Transpose tensor axes."""
        xp = get_backend_module()
        if axes is None:
            data = xp.transpose(self._data)
            labels = self._labels[::-1] if self._labels else None
        else:
            data = xp.transpose(self._data, axes)
            labels = tuple(self._labels[i] for i in axes) if self._labels else None
        return Tensor(data, backend=self._backend, labels=labels)
    
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Reshape the tensor."""
        xp = get_backend_module()
        data = xp.reshape(self._data, shape)
        return Tensor(data, backend=self._backend)
    
    def flatten(self) -> 'Tensor':
        """Flatten the tensor to 1D."""
        return self.reshape((-1,))
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove single-dimensional entries from the shape."""
        xp = get_backend_module()
        data = xp.squeeze(self._data, axis=axis)
        return Tensor(data, backend=self._backend)
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        """Add a dimension at the specified axis."""
        xp = get_backend_module()
        data = xp.expand_dims(self._data, axis=axis)
        return Tensor(data, backend=self._backend)
    
    def trace(self, axis1: int = 0, axis2: int = 1) -> 'Tensor':
        """Compute trace over specified axes."""
        xp = get_backend_module()
        data = xp.trace(self._data, axis1=axis1, axis2=axis2)
        return Tensor(data, backend=self._backend)
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        xp = get_backend_module()
        data = xp.matmul(self._data, other._data)
        return Tensor(data, backend=self._backend)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Dot product."""
        xp = get_backend_module()
        data = xp.dot(self._data, other._data)
        return Tensor(data, backend=self._backend)
    
    def tensordot(
        self, 
        other: 'Tensor', 
        axes: Union[int, Tuple[Sequence[int], Sequence[int]]]
    ) -> 'Tensor':
        """Tensor dot product."""
        xp = get_backend_module()
        data = xp.tensordot(self._data, other._data, axes=axes)
        return Tensor(data, backend=self._backend)
    
    def outer(self, other: 'Tensor') -> 'Tensor':
        """Outer product."""
        xp = get_backend_module()
        data = xp.outer(self._data.flatten(), other._data.flatten())
        new_shape = self.dims + other.dims
        return Tensor(data.reshape(new_shape), backend=self._backend)
    
    # ==================== Reduction Operations ====================
    
    def sum(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None, 
        keepdims: bool = False
    ) -> 'Tensor':
        """Sum over axes."""
        xp = get_backend_module()
        data = xp.sum(self._data, axis=axis, keepdims=keepdims)
        return Tensor(data, backend=self._backend)
    
    def mean(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None, 
        keepdims: bool = False
    ) -> 'Tensor':
        """Mean over axes."""
        xp = get_backend_module()
        data = xp.mean(self._data, axis=axis, keepdims=keepdims)
        return Tensor(data, backend=self._backend)
    
    def max(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None, 
        keepdims: bool = False
    ) -> 'Tensor':
        """Maximum over axes."""
        xp = get_backend_module()
        data = xp.max(self._data, axis=axis, keepdims=keepdims)
        return Tensor(data, backend=self._backend)
    
    def min(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None, 
        keepdims: bool = False
    ) -> 'Tensor':
        """Minimum over axes."""
        xp = get_backend_module()
        data = xp.min(self._data, axis=axis, keepdims=keepdims)
        return Tensor(data, backend=self._backend)
    
    # ==================== Special Operations ====================
    
    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        xp = get_backend_module()
        return Tensor(xp.exp(self._data), backend=self._backend)
    
    def log(self) -> 'Tensor':
        """Element-wise natural logarithm."""
        xp = get_backend_module()
        return Tensor(xp.log(self._data), backend=self._backend)
    
    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        xp = get_backend_module()
        return Tensor(xp.sqrt(self._data), backend=self._backend)
    
    def abs_squared(self) -> 'Tensor':
        """Element-wise absolute value squared."""
        xp = get_backend_module()
        return Tensor(xp.abs(self._data) ** 2, backend=self._backend)
    
    def real(self) -> 'Tensor':
        """Real part of the tensor."""
        xp = get_backend_module()
        return Tensor(xp.real(self._data), backend=self._backend)
    
    def imag(self) -> 'Tensor':
        """Imaginary part of the tensor."""
        xp = get_backend_module()
        return Tensor(xp.imag(self._data), backend=self._backend)
    
    # ==================== Index/Slice Operations ====================
    
    def index_select(self, axis: int, indices: Union[List[int], np.ndarray]) -> 'Tensor':
        """Select entries along an axis."""
        xp = get_backend_module()
        data = xp.take(self._data, indices, axis=axis)
        return Tensor(data, backend=self._backend)
    
    def split(self, indices_or_sections: Union[int, Sequence[int]], axis: int = 0) -> List['Tensor']:
        """Split tensor along an axis."""
        xp = get_backend_module()
        arrays = xp.split(self._data, indices_or_sections, axis=axis)
        return [Tensor(arr, backend=self._backend) for arr in arrays]
    
    # ==================== In-place Operations ====================
    
    def fill_(self, value: Union[float, complex]) -> 'Tensor':
        """Fill tensor with a value in-place."""
        self._data.fill(value)
        return self
    
    def zero_(self) -> 'Tensor':
        """Zero out the tensor in-place."""
        return self.fill_(0)
    
    def add_(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        """In-place addition."""
        if isinstance(other, Tensor):
            self._data += other._data
        else:
            self._data += other
        return self
    
    def mul_(self, other: Union['Tensor', float, complex]) -> 'Tensor':
        """In-place multiplication."""
        if isinstance(other, Tensor):
            self._data *= other._data
        else:
            self._data *= other
        return self


def tensordot(
    a: Tensor, 
    b: Tensor, 
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]]
) -> Tensor:
    """Tensor dot product of two tensors."""
    return a.tensordot(b, axes)


def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """
    Einstein summation convention.
    
    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation
    operands : Tensor
        The tensors to operate on
    
    Returns
    -------
    Tensor
        The result of the einsum operation
    """
    xp = get_backend_module()
    arrays = [op.data for op in operands]
    result = xp.einsum(subscripts, *arrays)
    backend = operands[0].backend if operands else get_backend()
    return Tensor(result, backend=backend)
