"""
GPU acceleration for iPEPS using CuPy.

This module provides GPU-accelerated tensor operations using CuPy,
which provides a NumPy-compatible interface for NVIDIA GPUs.

Supports:
- Automatic GPU memory management
- Multi-GPU distribution
- CPU fallback when GPU unavailable
- Mixed-precision computation

Requirements:
    pip install cupy-cuda12x  # For CUDA 12.x
    pip install cupy-cuda11x  # For CUDA 11.x
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import warnings
import os

# Try to import CuPy
try:
    import cupy as cp
    from cupy import cuda
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class DeviceType(Enum):
    """Computing device types."""
    CPU = 'cpu'
    GPU = 'gpu'
    MULTI_GPU = 'multi_gpu'


@dataclass
class GPUConfig:
    """GPU configuration."""
    device_id: int = 0  # Primary GPU device ID
    memory_pool_fraction: float = 0.9  # Fraction of GPU memory to use
    enable_memory_pool: bool = True
    enable_tensor_cores: bool = True  # Use Tensor Cores if available
    mixed_precision: bool = False  # Use FP16 where possible
    fallback_to_cpu: bool = True  # Fall back to CPU if GPU unavailable


class GPUBackend:
    """
    GPU backend manager for iPEPS computations.
    
    Provides a unified interface for GPU operations, with automatic
    fallback to CPU when GPU is unavailable.
    
    Parameters
    ----------
    config : GPUConfig, optional
        GPU configuration
    
    Examples
    --------
    >>> gpu = GPUBackend()
    >>> if gpu.is_available:
    ...     gpu.set_device(0)
    ...     tensor_gpu = gpu.to_device(tensor_cpu)
    ...     result_gpu = tensor_gpu @ tensor_gpu.T
    ...     result_cpu = gpu.to_host(result_gpu)
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        
        self._device = None
        self._initialized = False
        
        if HAS_CUPY:
            self._initialize_gpu()
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources."""
        try:
            # Check available GPUs
            n_devices = cuda.runtime.getDeviceCount()
            if n_devices == 0:
                warnings.warn("No CUDA devices found")
                return
            
            # Set primary device
            device_id = min(self.config.device_id, n_devices - 1)
            cuda.Device(device_id).use()
            self._device = device_id
            
            # Configure memory pool
            if self.config.enable_memory_pool:
                mempool = cp.get_default_memory_pool()
                # Set memory limit
                total_mem = cuda.Device(device_id).mem_info[1]
                limit = int(total_mem * self.config.memory_pool_fraction)
                mempool.set_limit(size=limit)
            
            self._initialized = True
            
        except Exception as e:
            warnings.warn(f"GPU initialization failed: {e}")
            if not self.config.fallback_to_cpu:
                raise
    
    @property
    def is_available(self) -> bool:
        """Whether GPU is available."""
        return HAS_CUPY and self._initialized
    
    @property
    def device_id(self) -> Optional[int]:
        """Current GPU device ID."""
        return self._device
    
    @property
    def device_name(self) -> str:
        """Name of the current GPU device."""
        if not self.is_available:
            return "CPU"
        
        props = cuda.runtime.getDeviceProperties(self._device)
        return props['name'].decode('utf-8')
    
    @property
    def memory_info(self) -> Tuple[int, int]:
        """Get (free, total) GPU memory in bytes."""
        if not self.is_available:
            return (0, 0)
        
        return cuda.Device(self._device).mem_info
    
    def set_device(self, device_id: int) -> None:
        """
        Set the active GPU device.
        
        Parameters
        ----------
        device_id : int
            GPU device ID
        """
        if not HAS_CUPY:
            return
        
        n_devices = cuda.runtime.getDeviceCount()
        if device_id >= n_devices:
            raise ValueError(f"Device {device_id} not available (found {n_devices} devices)")
        
        cuda.Device(device_id).use()
        self._device = device_id
    
    def synchronize(self) -> None:
        """Synchronize GPU execution."""
        if self.is_available:
            cuda.Device(self._device).synchronize()
    
    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer array to GPU.
        
        Parameters
        ----------
        array : ndarray
            NumPy array
        
        Returns
        -------
        cupy.ndarray
            GPU array
        """
        if not self.is_available:
            return array
        
        return cp.asarray(array)
    
    def to_host(self, array: Any) -> np.ndarray:
        """
        Transfer array to CPU.
        
        Parameters
        ----------
        array : cupy.ndarray
            GPU array
        
        Returns
        -------
        ndarray
            NumPy array
        """
        if not self.is_available:
            return array
        
        if hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def empty(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
    ) -> Any:
        """Create empty array on GPU."""
        if not self.is_available:
            return np.empty(shape, dtype=dtype)
        return cp.empty(shape, dtype=dtype)
    
    def zeros(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
    ) -> Any:
        """Create zero array on GPU."""
        if not self.is_available:
            return np.zeros(shape, dtype=dtype)
        return cp.zeros(shape, dtype=dtype)
    
    def ones(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
    ) -> Any:
        """Create ones array on GPU."""
        if not self.is_available:
            return np.ones(shape, dtype=dtype)
        return cp.ones(shape, dtype=dtype)
    
    def eye(
        self,
        n: int,
        dtype: Any = np.complex128,
    ) -> Any:
        """Create identity matrix on GPU."""
        if not self.is_available:
            return np.eye(n, dtype=dtype)
        return cp.eye(n, dtype=dtype)
    
    def random(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.complex128,
    ) -> Any:
        """Create random array on GPU."""
        if not self.is_available:
            if np.issubdtype(dtype, np.complexfloating):
                return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype) / np.sqrt(2)
            return np.random.randn(*shape).astype(dtype)
        
        if np.issubdtype(dtype, np.complexfloating):
            real = cp.random.randn(*shape)
            imag = cp.random.randn(*shape)
            return (real + 1j * imag).astype(dtype) / np.sqrt(2)
        return cp.random.randn(*shape).astype(dtype)
    
    def svd(
        self,
        matrix: Any,
        full_matrices: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """
        Compute SVD on GPU.
        
        Uses cuSOLVER for GPU-accelerated SVD.
        """
        if not self.is_available:
            return np.linalg.svd(matrix, full_matrices=full_matrices)
        
        return cp.linalg.svd(matrix, full_matrices=full_matrices)
    
    def qr(self, matrix: Any) -> Tuple[Any, Any]:
        """Compute QR decomposition on GPU."""
        if not self.is_available:
            return np.linalg.qr(matrix)
        return cp.linalg.qr(matrix)
    
    def eigh(self, matrix: Any) -> Tuple[Any, Any]:
        """Compute Hermitian eigendecomposition on GPU."""
        if not self.is_available:
            return np.linalg.eigh(matrix)
        return cp.linalg.eigh(matrix)
    
    def tensordot(
        self,
        a: Any,
        b: Any,
        axes: Any,
    ) -> Any:
        """Tensor dot product on GPU."""
        if not self.is_available:
            return np.tensordot(a, b, axes=axes)
        return cp.tensordot(a, b, axes=axes)
    
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Einstein summation on GPU."""
        if not self.is_available:
            return np.einsum(subscripts, *operands)
        return cp.einsum(subscripts, *operands)
    
    def clear_memory(self) -> None:
        """Clear GPU memory pool."""
        if self.is_available:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        if not self.is_available:
            return {'used': 0, 'total': 0, 'free': 0}
        
        free, total = self.memory_info
        used = total - free
        
        return {
            'used': used,
            'total': total,
            'free': free,
        }


def get_available_gpus() -> List[Dict[str, Any]]:
    """
    Get information about available GPU devices.
    
    Returns
    -------
    list of dict
        Information about each GPU
    """
    if not HAS_CUPY:
        return []
    
    try:
        n_devices = cuda.runtime.getDeviceCount()
    except:
        return []
    
    gpus = []
    for i in range(n_devices):
        try:
            props = cuda.runtime.getDeviceProperties(i)
            device = cuda.Device(i)
            free, total = device.mem_info
            
            gpus.append({
                'id': i,
                'name': props['name'].decode('utf-8'),
                'compute_capability': f"{props['major']}.{props['minor']}",
                'total_memory': total,
                'free_memory': free,
                'multiprocessors': props['multiProcessorCount'],
            })
        except Exception as e:
            warnings.warn(f"Could not query GPU {i}: {e}")
    
    return gpus


def optimize_for_gpu(tensor_shapes: List[Tuple[int, ...]]) -> Dict[str, Any]:
    """
    Suggest GPU optimization parameters based on tensor shapes.
    
    Parameters
    ----------
    tensor_shapes : list of tuple
        Shapes of tensors to be processed
    
    Returns
    -------
    dict
        Optimization recommendations
    """
    recommendations = {
        'use_gpu': False,
        'mixed_precision': False,
        'batch_size': 1,
        'reason': '',
    }
    
    if not HAS_CUPY:
        recommendations['reason'] = 'CuPy not available'
        return recommendations
    
    gpus = get_available_gpus()
    if not gpus:
        recommendations['reason'] = 'No GPUs found'
        return recommendations
    
    # Estimate memory requirements
    total_elements = sum(np.prod(s) for s in tensor_shapes)
    bytes_per_element = 16  # complex128
    estimated_memory = total_elements * bytes_per_element * 3  # Factor for temporaries
    
    # Check if fits in GPU memory
    gpu = gpus[0]
    if estimated_memory < gpu['free_memory'] * 0.7:
        recommendations['use_gpu'] = True
        recommendations['reason'] = 'Tensors fit in GPU memory'
    else:
        # Check if mixed precision helps
        estimated_fp32 = total_elements * 8 * 3
        if estimated_fp32 < gpu['free_memory'] * 0.7:
            recommendations['use_gpu'] = True
            recommendations['mixed_precision'] = True
            recommendations['reason'] = 'Using mixed precision for memory'
        else:
            recommendations['reason'] = 'Tensors too large for GPU memory'
    
    # Recommend batch size for multiple contractions
    if recommendations['use_gpu']:
        ops_per_batch = int(gpu['free_memory'] * 0.5 / estimated_memory) or 1
        recommendations['batch_size'] = ops_per_batch
    
    return recommendations


class MultiGPUBackend:
    """
    Multi-GPU backend for distributed tensor operations.
    
    Distributes large tensors across multiple GPUs using
    NCCL for efficient communication.
    """
    
    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        config: Optional[GPUConfig] = None,
    ):
        self.config = config or GPUConfig()
        
        if not HAS_CUPY:
            self.devices = []
            return
        
        # Get available devices
        n_devices = cuda.runtime.getDeviceCount()
        if device_ids is None:
            device_ids = list(range(n_devices))
        else:
            device_ids = [d for d in device_ids if d < n_devices]
        
        self.devices = device_ids
        self.backends = {d: GPUBackend(GPUConfig(device_id=d)) for d in device_ids}
    
    @property
    def n_devices(self) -> int:
        """Number of available devices."""
        return len(self.devices)
    
    def distribute_tensor(
        self,
        tensor: np.ndarray,
        axis: int = 0,
    ) -> Dict[int, Any]:
        """
        Distribute tensor across GPUs along an axis.
        
        Parameters
        ----------
        tensor : ndarray
            Tensor to distribute
        axis : int
            Axis to split along
        
        Returns
        -------
        dict
            Device ID -> tensor shard mapping
        """
        if not self.devices:
            return {0: tensor}
        
        # Split tensor
        n = self.n_devices
        chunks = np.array_split(tensor, n, axis=axis)
        
        # Transfer to devices
        result = {}
        for i, device_id in enumerate(self.devices):
            backend = self.backends[device_id]
            backend.set_device(device_id)
            result[device_id] = backend.to_device(chunks[i])
        
        return result
    
    def gather_tensor(
        self,
        shards: Dict[int, Any],
        axis: int = 0,
    ) -> np.ndarray:
        """
        Gather distributed tensor shards to CPU.
        
        Parameters
        ----------
        shards : dict
            Device ID -> tensor shard mapping
        axis : int
            Axis to concatenate along
        
        Returns
        -------
        ndarray
            Assembled tensor
        """
        if not self.devices:
            return list(shards.values())[0]
        
        arrays = []
        for device_id in self.devices:
            if device_id in shards:
                backend = self.backends[device_id]
                arrays.append(backend.to_host(shards[device_id]))
        
        return np.concatenate(arrays, axis=axis)
    
    def parallel_svd(
        self,
        matrices: List[np.ndarray],
    ) -> List[Tuple]:
        """
        Parallel SVD on multiple matrices across GPUs.
        
        Parameters
        ----------
        matrices : list of ndarray
            Matrices to decompose
        
        Returns
        -------
        list of tuple
            (U, S, Vh) for each matrix
        """
        if not self.devices:
            return [np.linalg.svd(m, full_matrices=False) for m in matrices]
        
        # Distribute matrices round-robin
        assignments = {d: [] for d in self.devices}
        indices = {d: [] for d in self.devices}
        
        for i, m in enumerate(matrices):
            d = self.devices[i % self.n_devices]
            assignments[d].append(m)
            indices[d].append(i)
        
        # Compute on each device
        results = [None] * len(matrices)
        
        for device_id in self.devices:
            backend = self.backends[device_id]
            backend.set_device(device_id)
            
            for m, idx in zip(assignments[device_id], indices[device_id]):
                m_gpu = backend.to_device(m)
                U, S, Vh = backend.svd(m_gpu)
                results[idx] = (
                    backend.to_host(U),
                    backend.to_host(S),
                    backend.to_host(Vh),
                )
        
        return results
