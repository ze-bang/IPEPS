"""
MPI parallelization for iPEPS.

This module provides MPI-based parallelization for:
- Distributing CTMRG moves across processes
- Parallel bond updates
- Distributed observable computation

Uses mpi4py for communication.

Example usage:
    mpirun -np 16 python -m ipeps.run --config config.yaml
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import warnings

# Try to import mpi4py
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


@dataclass
class MPIConfig:
    """MPI configuration."""
    comm: Any = None  # MPI communicator
    root: int = 0  # Root process rank
    enable_logging: bool = True


class MPIManager:
    """
    Manager for MPI parallelization.
    
    Handles initialization, communication, and synchronization
    for distributed iPEPS calculations.
    
    Parameters
    ----------
    config : MPIConfig, optional
        MPI configuration
    
    Examples
    --------
    >>> mpi = MPIManager()
    >>> if mpi.is_root:
    ...     print(f"Running on {mpi.size} processes")
    >>> # Distribute work
    >>> local_bonds = mpi.scatter_bonds(all_bonds)
    >>> # Compute locally
    >>> local_results = [compute(b) for b in local_bonds]
    >>> # Gather results
    >>> all_results = mpi.gather(local_results)
    """
    
    def __init__(self, config: Optional[MPIConfig] = None):
        self.config = config or MPIConfig()
        
        if HAS_MPI:
            self.comm = self.config.comm or MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        self._initialized = True
    
    @property
    def is_root(self) -> bool:
        """Whether this is the root process."""
        return self.rank == self.config.root
    
    @property
    def is_parallel(self) -> bool:
        """Whether running in parallel mode."""
        return self.size > 1
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.comm is not None:
            self.comm.Barrier()
    
    def broadcast(self, data: Any, root: Optional[int] = None) -> Any:
        """
        Broadcast data from root to all processes.
        
        Parameters
        ----------
        data : any
            Data to broadcast (only significant on root)
        root : int, optional
            Root process rank (default: config.root)
        
        Returns
        -------
        any
            The broadcast data on all processes
        """
        if self.comm is None:
            return data
        
        root = root if root is not None else self.config.root
        return self.comm.bcast(data, root=root)
    
    def scatter(self, data: List[Any], root: Optional[int] = None) -> Any:
        """
        Scatter a list of data to all processes.
        
        Parameters
        ----------
        data : list
            List of items to distribute (on root) or None (on others)
        root : int, optional
            Root process rank
        
        Returns
        -------
        any
            Local portion of the scattered data
        """
        if self.comm is None:
            return data[0] if data else None
        
        root = root if root is not None else self.config.root
        return self.comm.scatter(data, root=root)
    
    def gather(self, data: Any, root: Optional[int] = None) -> Optional[List[Any]]:
        """
        Gather data from all processes to root.
        
        Parameters
        ----------
        data : any
            Local data to gather
        root : int, optional
            Root process rank
        
        Returns
        -------
        list or None
            Gathered data on root, None on other processes
        """
        if self.comm is None:
            return [data]
        
        root = root if root is not None else self.config.root
        return self.comm.gather(data, root=root)
    
    def allgather(self, data: Any) -> List[Any]:
        """
        Gather data from all processes to all processes.
        
        Parameters
        ----------
        data : any
            Local data to gather
        
        Returns
        -------
        list
            Gathered data on all processes
        """
        if self.comm is None:
            return [data]
        
        return self.comm.allgather(data)
    
    def reduce(
        self,
        data: np.ndarray,
        op: str = 'sum',
        root: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Reduce data from all processes to root.
        
        Parameters
        ----------
        data : ndarray
            Local data to reduce
        op : str
            Reduction operation ('sum', 'max', 'min', 'prod')
        root : int, optional
            Root process rank
        
        Returns
        -------
        ndarray or None
            Reduced data on root, None on other processes
        """
        if self.comm is None:
            return data
        
        root = root if root is not None else self.config.root
        
        op_map = {
            'sum': MPI.SUM,
            'max': MPI.MAX,
            'min': MPI.MIN,
            'prod': MPI.PROD,
        }
        mpi_op = op_map.get(op, MPI.SUM)
        
        result = np.zeros_like(data) if self.rank == root else None
        self.comm.Reduce(data, result, op=mpi_op, root=root)
        
        return result
    
    def allreduce(self, data: np.ndarray, op: str = 'sum') -> np.ndarray:
        """
        Reduce data from all processes to all processes.
        
        Parameters
        ----------
        data : ndarray
            Local data to reduce
        op : str
            Reduction operation
        
        Returns
        -------
        ndarray
            Reduced data on all processes
        """
        if self.comm is None:
            return data
        
        op_map = {
            'sum': MPI.SUM,
            'max': MPI.MAX,
            'min': MPI.MIN,
            'prod': MPI.PROD,
        }
        mpi_op = op_map.get(op, MPI.SUM)
        
        result = np.zeros_like(data)
        self.comm.Allreduce(data, result, op=mpi_op)
        
        return result
    
    def send(self, data: Any, dest: int, tag: int = 0) -> None:
        """Send data to a specific process."""
        if self.comm is not None:
            self.comm.send(data, dest=dest, tag=tag)
    
    def recv(self, source: int, tag: int = 0) -> Any:
        """Receive data from a specific process."""
        if self.comm is not None:
            return self.comm.recv(source=source, tag=tag)
        return None
    
    def scatter_bonds(
        self,
        bonds: List[Tuple],
    ) -> List[Tuple]:
        """
        Distribute bonds evenly across processes.
        
        Parameters
        ----------
        bonds : list
            List of bond tuples (on root)
        
        Returns
        -------
        list
            Local subset of bonds for this process
        """
        if not self.is_parallel:
            return bonds
        
        if self.is_root:
            # Distribute as evenly as possible
            chunks = [[] for _ in range(self.size)]
            for i, bond in enumerate(bonds):
                chunks[i % self.size].append(bond)
        else:
            chunks = None
        
        return self.scatter(chunks)
    
    def parallel_map(
        self,
        func: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Apply a function to items in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply
        items : list
            Items to process (on root)
        
        Returns
        -------
        list
            Results from all processes (on root)
        """
        # Scatter items
        local_items = self.scatter_bonds(items) if self.is_root else self.scatter(None)
        
        # Process locally
        local_results = [func(item) for item in local_items]
        
        # Gather results
        all_results = self.gather(local_results)
        
        if self.is_root:
            # Flatten
            return [r for chunk in all_results for r in chunk]
        return None
    
    def log(self, message: str) -> None:
        """Log a message (only from root if enabled)."""
        if self.config.enable_logging and self.is_root:
            print(message)


def distribute_bonds(
    bonds: List[Tuple],
    mpi_manager: Optional[MPIManager] = None,
) -> List[Tuple]:
    """
    Distribute bonds across MPI processes.
    
    Parameters
    ----------
    bonds : list
        All bonds to distribute
    mpi_manager : MPIManager, optional
        MPI manager instance
    
    Returns
    -------
    list
        Local bonds for this process
    """
    if mpi_manager is None:
        mpi_manager = MPIManager()
    
    return mpi_manager.scatter_bonds(bonds)


def gather_results(
    local_results: List[Any],
    mpi_manager: Optional[MPIManager] = None,
) -> Optional[List[Any]]:
    """
    Gather results from all MPI processes.
    
    Parameters
    ----------
    local_results : list
        Results from this process
    mpi_manager : MPIManager, optional
        MPI manager instance
    
    Returns
    -------
    list or None
        All results on root, None on other processes
    """
    if mpi_manager is None:
        mpi_manager = MPIManager()
    
    return mpi_manager.gather(local_results)


class MPIParallelCTMRG:
    """
    MPI-parallelized CTMRG.
    
    Distributes CTM moves across processes for faster convergence.
    Each process handles a subset of unit cell positions.
    """
    
    def __init__(
        self,
        peps: Any,
        mpi_manager: Optional[MPIManager] = None,
    ):
        self.peps = peps
        self.mpi = mpi_manager or MPIManager()
        
        # Distribute unit cell sites
        all_sites = list(peps.lattice.sites)
        self.local_sites = self._distribute_sites(all_sites)
    
    def _distribute_sites(self, sites: List) -> List:
        """Distribute sites across processes."""
        if not self.mpi.is_parallel:
            return sites
        
        # Simple round-robin distribution
        return sites[self.mpi.rank::self.mpi.size]
    
    def run_parallel_move(self, direction: str) -> float:
        """
        Run CTM move in parallel.
        
        Each process updates its local sites, then results are synchronized.
        """
        # Local computation
        local_error = 0.0
        local_updates = {}
        
        for site in self.local_sites:
            # Compute local CTM update
            # (implementation depends on CTMRG internals)
            pass
        
        # Synchronize updates
        all_updates = self.mpi.allgather(local_updates)
        total_error = self.mpi.allreduce(np.array([local_error]), op='sum')[0]
        
        # Apply updates
        for updates in all_updates:
            for key, value in updates.items():
                # Apply to local state
                pass
        
        return total_error


class MPIParallelUpdate:
    """
    MPI-parallelized tensor updates.
    
    Distributes bond updates across processes for Simple/Full Update.
    """
    
    def __init__(
        self,
        peps: Any,
        mpi_manager: Optional[MPIManager] = None,
    ):
        self.peps = peps
        self.mpi = mpi_manager or MPIManager()
    
    def parallel_simple_update_step(
        self,
        gates: List[Tuple],
    ) -> float:
        """
        Perform one parallel Simple Update step.
        
        Gates are distributed across processes, applied independently,
        then synchronized.
        """
        # For Simple Update, bonds can be updated in parallel
        # if they don't share vertices
        
        # Color bonds to find independent sets
        independent_sets = self._color_bonds(gates)
        
        total_error = 0.0
        
        for gate_set in independent_sets:
            # Distribute this set
            local_gates = self.mpi.scatter_bonds(gate_set)
            
            # Apply locally
            local_errors = []
            for gate in local_gates:
                # Apply gate (from SimpleUpdate)
                error = 0.0  # Placeholder
                local_errors.append(error)
            
            # Gather errors
            all_errors = self.mpi.allgather(sum(local_errors))
            total_error += sum(all_errors)
            
            # Synchronize tensor updates
            self._sync_tensors()
        
        return total_error
    
    def _color_bonds(
        self,
        gates: List[Tuple],
    ) -> List[List[Tuple]]:
        """
        Group bonds by color so that same-colored bonds don't share sites.
        """
        # Simple greedy coloring
        colors = {}
        max_color = 0
        
        for gate in gates:
            site1, site2 = gate[0], gate[1]
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for other, color in colors.items():
                if other[0] in (site1, site2) or other[1] in (site1, site2):
                    neighbor_colors.add(color)
            
            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            colors[(site1, site2)] = color
            max_color = max(max_color, color)
        
        # Group by color
        groups = [[] for _ in range(max_color + 1)]
        for gate in gates:
            site1, site2 = gate[0], gate[1]
            color = colors[(site1, site2)]
            groups[color].append(gate)
        
        return groups
    
    def _sync_tensors(self) -> None:
        """Synchronize tensor updates across processes."""
        # Broadcast updated tensors from each process
        for rank in range(self.mpi.size):
            # Each process broadcasts its local updates
            if rank == self.mpi.rank:
                updates = {pos: self.peps.tensors[pos].numpy() 
                          for pos in self.peps.tensors}
            else:
                updates = None
            
            updates = self.mpi.broadcast(updates, root=rank)
            
            # Apply updates from this process
            if rank != self.mpi.rank:
                for pos, data in updates.items():
                    if pos in self.peps.tensors:
                        from ipeps.core.tensor import Tensor
                        self.peps.tensors[pos] = Tensor(data)
