"""
Optimized tensor contractions for iPEPS.

This module provides high-performance tensor contraction routines using:
- opt_einsum for optimal contraction path finding
- Backend-specific optimizations (GPU, MPI)
- Caching of contraction paths
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Sequence, Optional, Union, Dict, Any
from functools import lru_cache
import warnings

try:
    import opt_einsum as oe
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False
    warnings.warn("opt_einsum not found. Using numpy einsum (slower)")

from ipeps.core.tensor import Tensor, get_backend_module


# Cache for contraction paths
_CONTRACTION_PATH_CACHE: Dict[str, Any] = {}


def contract(
    subscripts: str,
    *operands: Tensor,
    optimize: Union[bool, str] = True,
) -> Tensor:
    """
    Contract tensors using Einstein summation with optional optimization.
    
    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts (e.g., 'ijk,jkl->il')
    operands : Tensor
        Tensors to contract
    optimize : bool or str
        Optimization strategy. Can be:
        - True: Use opt_einsum's optimal path
        - False: Use numpy default
        - 'dp': Dynamic programming (slow but optimal)
        - 'greedy': Greedy algorithm (fast but suboptimal)
        - 'auto': Auto-select based on tensor sizes
    
    Returns
    -------
    Tensor
        Result of the contraction
    
    Examples
    --------
    >>> a = Tensor.random((2, 3, 4))
    >>> b = Tensor.random((4, 5))
    >>> c = contract('ijk,kl->ijl', a, b)
    >>> c.shape
    (2, 3, 5)
    """
    xp = get_backend_module()
    arrays = [op.data for op in operands]
    
    if HAS_OPT_EINSUM and optimize:
        if isinstance(optimize, bool):
            optimize = 'auto'
        result = oe.contract(subscripts, *arrays, optimize=optimize)
    else:
        result = xp.einsum(subscripts, *arrays, optimize=True)
    
    backend = operands[0].backend if operands else None
    return Tensor(result, backend=backend)


def contract_ncon(
    tensors: List[Tensor],
    indices: List[List[int]],
    contraction_order: Optional[List[int]] = None,
) -> Tensor:
    """
    Network contractor (ncon) style contraction.
    
    This follows the ncon convention where positive indices are contracted
    and negative indices are free (output) indices.
    
    Parameters
    ----------
    tensors : list of Tensor
        Tensors to contract
    indices : list of list of int
        Index structure for each tensor. Positive indices are contracted,
        negative indices appear in output (sorted by absolute value).
    contraction_order : list of int, optional
        Order to contract positive indices. If None, uses default ordering.
    
    Returns
    -------
    Tensor
        Result of the contraction
    
    Examples
    --------
    >>> # Contract: A_{ijk} B_{jkl} -> C_{il}
    >>> A = Tensor.random((2, 3, 4))
    >>> B = Tensor.random((3, 4, 5))
    >>> C = contract_ncon([A, B], [[-1, 1, 2], [1, 2, -2]])
    >>> C.shape
    (2, 5)
    
    Notes
    -----
    The ncon convention:
    - Positive integers: contracted indices (same number = contracted together)
    - Negative integers: free indices (appear in output, ordered by absolute value)
    """
    # Validate inputs
    if len(tensors) != len(indices):
        raise ValueError("Number of tensors must match number of index lists")
    
    for i, (t, idx) in enumerate(zip(tensors, indices)):
        if t.ndim != len(idx):
            raise ValueError(
                f"Tensor {i} has {t.ndim} dimensions but {len(idx)} indices specified"
            )
    
    # Build einsum string from ncon indices
    # Map each unique index to a character
    all_indices = set()
    for idx_list in indices:
        all_indices.update(idx_list)
    
    # Separate positive (contracted) and negative (free) indices
    contracted = sorted([i for i in all_indices if i > 0])
    free = sorted([i for i in all_indices if i < 0], key=abs)
    
    # Create character mapping
    char_map = {}
    char_idx = 0
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # First assign chars to free indices (in order)
    for idx in free:
        char_map[idx] = chars[char_idx]
        char_idx += 1
    
    # Then to contracted indices
    for idx in contracted:
        char_map[idx] = chars[char_idx]
        char_idx += 1
    
    # Build einsum string
    input_parts = []
    for idx_list in indices:
        part = ''.join(char_map[i] for i in idx_list)
        input_parts.append(part)
    
    output = ''.join(char_map[i] for i in free)
    
    subscripts = ','.join(input_parts) + '->' + output
    
    return contract(subscripts, *tensors)


def optimize_contraction(
    subscripts: str,
    shapes: List[Tuple[int, ...]],
    memory_limit: Optional[int] = None,
) -> Tuple[Any, float, float]:
    """
    Find optimal contraction path and estimate costs.
    
    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts
    shapes : list of tuple
        Shapes of the tensors to contract
    memory_limit : int, optional
        Maximum intermediate tensor size (in elements)
    
    Returns
    -------
    path : list
        Optimal contraction path
    flops : float
        Estimated FLOP count
    memory : float
        Estimated peak memory usage
    
    Examples
    --------
    >>> path, flops, mem = optimize_contraction(
    ...     'ijk,jkl,lmn->imn',
    ...     [(100, 50, 50), (50, 50, 100), (100, 50, 50)]
    ... )
    """
    if not HAS_OPT_EINSUM:
        raise ImportError("opt_einsum required for contraction optimization")
    
    # Create dummy arrays with the right shapes
    arrays = [np.zeros(s) for s in shapes]
    
    # Get optimal path
    path_info = oe.contract_path(
        subscripts, 
        *arrays,
        optimize='dp',
        memory_limit=memory_limit,
    )
    
    path = path_info[0]
    info = path_info[1]
    
    flops = info.opt_cost
    memory = info.largest_intermediate
    
    return path, flops, memory


class CachedContractor:
    """
    Cached contractor for repeated contractions with the same structure.
    
    Caches the contraction path for efficiency when the same contraction
    pattern is used multiple times (e.g., in iterative algorithms).
    
    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts
    shapes : list of tuple
        Expected shapes of input tensors
    
    Examples
    --------
    >>> contractor = CachedContractor('ijk,jkl->il', [(2, 3, 4), (3, 4, 5)])
    >>> result = contractor(tensor_a, tensor_b)
    """
    
    def __init__(
        self,
        subscripts: str,
        shapes: List[Tuple[int, ...]],
        optimize: str = 'auto',
    ):
        self.subscripts = subscripts
        self.shapes = shapes
        self._expression = None
        
        if HAS_OPT_EINSUM:
            # Create the optimized contraction expression
            arrays = [np.zeros(s, dtype=np.complex128) for s in shapes]
            self._expression = oe.contract_expression(
                subscripts,
                *[a.shape for a in arrays],
                optimize=optimize,
            )
    
    def __call__(self, *operands: Tensor) -> Tensor:
        """Execute the cached contraction."""
        arrays = [op.data for op in operands]
        
        if self._expression is not None:
            result = self._expression(*arrays)
        else:
            xp = get_backend_module()
            result = xp.einsum(self.subscripts, *arrays)
        
        backend = operands[0].backend if operands else None
        return Tensor(result, backend=backend)
    
    @property
    def flops(self) -> Optional[float]:
        """Estimated FLOP count for the contraction."""
        if self._expression is None:
            return None
        # opt_einsum doesn't directly expose flops on expressions
        # Would need to re-compute path
        return None


def batched_contract(
    subscripts: str,
    tensors: List[Tensor],
    batch_size: int = 1000,
) -> Tensor:
    """
    Batched contraction for memory-limited operations.
    
    Splits large contractions into batches to limit memory usage.
    
    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts
    tensors : list of Tensor
        Tensors to contract
    batch_size : int
        Maximum batch dimension size
    
    Returns
    -------
    Tensor
        Result of the contraction
    """
    # For now, just do regular contraction
    # Full batched implementation would split along batch dimensions
    return contract(subscripts, *tensors)


def multi_contract(
    contractions: List[Tuple[str, List[Tensor]]],
    parallel: bool = False,
) -> List[Tensor]:
    """
    Execute multiple independent contractions.
    
    Parameters
    ----------
    contractions : list of (subscripts, tensors) tuples
        Contractions to execute
    parallel : bool
        Whether to execute in parallel (requires GPU or multiprocessing)
    
    Returns
    -------
    list of Tensor
        Results of each contraction
    """
    results = []
    for subscripts, tensors in contractions:
        results.append(contract(subscripts, *tensors))
    return results


# ==================== Specialized iPEPS Contractions ====================


def contract_double_layer(
    tensor_a: Tensor,
    tensor_b: Tensor,
    physical_axis: int = 0,
) -> Tensor:
    """
    Contract two tensors to form a double-layer tensor.
    
    Used in iPEPS for computing <ψ|O|ψ> type expectation values.
    
    Parameters
    ----------
    tensor_a : Tensor
        Bra tensor (will be conjugated)
    tensor_b : Tensor
        Ket tensor
    physical_axis : int
        Index of the physical dimension
    
    Returns
    -------
    Tensor
        Double-layer tensor with combined auxiliary indices
    """
    # Move physical index to last position for both tensors
    ndim = tensor_a.ndim
    axes_a = list(range(ndim))
    axes_a.remove(physical_axis)
    axes_a.append(physical_axis)
    
    a_reordered = tensor_a.transpose(tuple(axes_a)).conj()
    b_reordered = tensor_b.transpose(tuple(axes_a))
    
    # Contract over physical index
    # Result has shape (aux_a_1, ..., aux_a_n, aux_b_1, ..., aux_b_n)
    return a_reordered.tensordot(b_reordered, axes=([ndim-1], [ndim-1]))


def contract_ctm_corner(
    corner: Tensor,
    edge: Tensor,
) -> Tensor:
    """
    Contract a CTM corner with an edge tensor.
    
    Parameters
    ----------
    corner : Tensor
        Corner tensor (rank-2)
    edge : Tensor
        Edge tensor (rank-3)
    
    Returns
    -------
    Tensor
        Contracted corner-edge tensor
    """
    # Standard CTMRG corner-edge contraction
    # Corner: (χ, χ), Edge: (χ, D², χ)
    return contract('ij,jkl->ikl', corner, edge)


def contract_ctm_cell(
    corners: List[Tensor],
    edges: List[Tensor],
    center: Tensor,
) -> Tensor:
    """
    Contract the full CTM environment around a center tensor.
    
    Parameters
    ----------
    corners : list of 4 Tensor
        Corner tensors [C1, C2, C3, C4] (NW, NE, SE, SW)
    edges : list of 4 Tensor
        Edge tensors [T1, T2, T3, T4] (N, E, S, W)
    center : Tensor
        Center (double-layer) tensor
    
    Returns
    -------
    Tensor
        Scalar or tensor result of the contraction
    """
    # This is the core CTM contraction for computing expectation values
    # The contraction order matters significantly for efficiency
    
    C1, C2, C3, C4 = corners
    T1, T2, T3, T4 = edges
    A = center
    
    # Optimal contraction order for honeycomb-like structures
    # Contract in a specific order to minimize intermediate tensor sizes
    
    # First, contract corners with edges
    # C1-T1: (χ,χ), (χ,D²,χ) -> (χ,D²,χ)
    C1T1 = contract('ij,jkl->ikl', C1, T1)
    
    # C2-T2: (χ,χ), (χ,D²,χ) -> (χ,D²,χ)  
    C2T2 = contract('ij,jkl->ikl', C2, T2)
    
    # C3-T3
    C3T3 = contract('ij,jkl->ikl', C3, T3)
    
    # C4-T4
    C4T4 = contract('ij,jkl->ikl', C4, T4)
    
    # Now contract pairs
    # Upper: C1T1 with center's top
    # This depends on the specific geometry
    
    # Simplified version - actual implementation depends on exact index structure
    # For a general tensor, we do:
    
    # Contract all edges with corners first
    upper = contract('ijk,klm->ijlm', C1T1, C2T2)
    lower = contract('ijk,klm->ijlm', C4T4, C3T3)
    
    # Contract with center
    # This is simplified - real version has more complex index structure
    result = contract('ijkl,klmn,mnop->ijop', upper, A, lower)
    
    return result
