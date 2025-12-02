"""
Checkpointing and restart capabilities for iPEPS simulations.

Provides:
- State serialization to disk
- Automatic periodic checkpointing
- Graceful restart from checkpoints
- Version-aware checkpoint format
"""

from __future__ import annotations

import numpy as np
import json
import pickle
import gzip
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, asdict
import warnings


CHECKPOINT_VERSION = "1.0.0"
CHECKPOINT_MAGIC = b"IPEPS_CKPT"


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    version: str
    timestamp: str
    iteration: int
    energy: Optional[float]
    convergence: Optional[float]
    bond_dim: int
    lattice_type: str
    model_type: str
    description: str
    checksum: str  # For integrity verification


class CheckpointManager:
    """
    Manage checkpoints for iPEPS simulations.
    
    Provides automatic checkpointing, rotation, and restart capabilities.
    
    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory to store checkpoints
    max_checkpoints : int
        Maximum number of checkpoints to keep
    compress : bool
        Whether to compress checkpoints
    
    Examples
    --------
    >>> ckpt = CheckpointManager("./checkpoints")
    >>> ckpt.save(ipeps_state, ctmrg_env, iteration=100, energy=-0.5)
    >>> # Later...
    >>> state, env, metadata = ckpt.load_latest()
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        compress: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compress = compress
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute MD5 checksum of data."""
        return hashlib.md5(data).hexdigest()
    
    def _get_checkpoint_path(self, iteration: int) -> Path:
        """Get path for checkpoint at given iteration."""
        ext = ".ckpt.gz" if self.compress else ".ckpt"
        return self.checkpoint_dir / f"checkpoint_{iteration:08d}{ext}"
    
    def _serialize_tensor(self, tensor: Any) -> Dict[str, Any]:
        """Serialize a tensor to a dictionary."""
        if hasattr(tensor, 'data'):
            # Our Tensor class
            arr = tensor.data
            if hasattr(arr, 'get'):  # CuPy array
                arr = arr.get()
            return {
                'type': 'Tensor',
                'data': arr.tolist(),
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
            }
        elif hasattr(tensor, 'get'):  # CuPy array
            arr = tensor.get()
            return {
                'type': 'ndarray',
                'data': arr.tolist(),
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
            }
        elif isinstance(tensor, np.ndarray):
            return {
                'type': 'ndarray',
                'data': tensor.tolist(),
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
            }
        else:
            return tensor
    
    def _deserialize_tensor(self, data: Dict[str, Any]) -> Any:
        """Deserialize a tensor from a dictionary."""
        if not isinstance(data, dict):
            return data
        
        tensor_type = data.get('type')
        
        if tensor_type == 'ndarray':
            return np.array(data['data'], dtype=data['dtype'])
        elif tensor_type == 'Tensor':
            # Import here to avoid circular imports
            from ..core.tensor import Tensor
            arr = np.array(data['data'], dtype=data['dtype'])
            return Tensor(arr)
        else:
            return data
    
    def _serialize_state(
        self,
        ipeps_state: Any,
        environment: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Serialize iPEPS state and environment."""
        state_data = {}
        
        # Serialize iPEPS tensors
        if hasattr(ipeps_state, 'tensors'):
            state_data['tensors'] = {
                str(key): self._serialize_tensor(t)
                for key, t in ipeps_state.tensors.items()
            }
        
        if hasattr(ipeps_state, 'lambdas'):
            state_data['lambdas'] = {
                str(key): self._serialize_tensor(l)
                for key, l in ipeps_state.lambdas.items()
            }
        
        if hasattr(ipeps_state, 'unit_cell'):
            # Serialize unit cell info
            uc = ipeps_state.unit_cell
            state_data['unit_cell'] = {
                'lx': uc.lx,
                'ly': uc.ly,
            }
        
        # Serialize environment if provided
        if environment is not None:
            env_data = {}
            
            if hasattr(environment, 'corners'):
                env_data['corners'] = {
                    str(key): self._serialize_tensor(c)
                    for key, c in environment.corners.items()
                }
            
            if hasattr(environment, 'edges'):
                env_data['edges'] = {
                    str(key): self._serialize_tensor(e)
                    for key, e in environment.edges.items()
                }
            
            if hasattr(environment, 'chi'):
                env_data['chi'] = environment.chi
            
            state_data['environment'] = env_data
        
        return state_data
    
    def _deserialize_state(
        self,
        data: Dict[str, Any],
    ) -> tuple:
        """Deserialize iPEPS state and environment."""
        # Import here to avoid circular imports
        from ..core.ipeps_state import iPEPSState
        
        state = iPEPSState.__new__(iPEPSState)
        
        # Deserialize tensors
        if 'tensors' in data:
            state.tensors = {
                eval(key): self._deserialize_tensor(t)
                for key, t in data['tensors'].items()
            }
        
        if 'lambdas' in data:
            state.lambdas = {
                eval(key): self._deserialize_tensor(l)
                for key, l in data['lambdas'].items()
            }
        
        # Deserialize environment
        environment = None
        if 'environment' in data:
            from ..algorithms.ctmrg import CTMRGEnvironment
            
            env_data = data['environment']
            environment = CTMRGEnvironment.__new__(CTMRGEnvironment)
            
            if 'corners' in env_data:
                environment.corners = {
                    eval(key): self._deserialize_tensor(c)
                    for key, c in env_data['corners'].items()
                }
            
            if 'edges' in env_data:
                environment.edges = {
                    eval(key): self._deserialize_tensor(e)
                    for key, e in env_data['edges'].items()
                }
            
            if 'chi' in env_data:
                environment.chi = env_data['chi']
        
        return state, environment
    
    def save(
        self,
        ipeps_state: Any,
        environment: Optional[Any] = None,
        iteration: int = 0,
        energy: Optional[float] = None,
        convergence: Optional[float] = None,
        description: str = "",
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint.
        
        Parameters
        ----------
        ipeps_state : iPEPSState
            iPEPS state to save
        environment : CTMRGEnvironment, optional
            CTMRG environment to save
        iteration : int
            Current iteration number
        energy : float, optional
            Current energy
        convergence : float, optional
            Current convergence measure
        description : str
            Human-readable description
        extra_data : dict, optional
            Additional data to save
        
        Returns
        -------
        Path
            Path to saved checkpoint
        """
        # Build checkpoint data
        checkpoint = {
            'state': self._serialize_state(ipeps_state, environment),
            'extra': extra_data or {},
        }
        
        # Serialize to bytes
        checkpoint_bytes = pickle.dumps(checkpoint)
        
        # Compute checksum
        checksum = self._compute_checksum(checkpoint_bytes)
        
        # Get metadata
        bond_dim = 0
        if hasattr(ipeps_state, 'tensors') and ipeps_state.tensors:
            first_tensor = list(ipeps_state.tensors.values())[0]
            if hasattr(first_tensor, 'data'):
                bond_dim = first_tensor.data.shape[0]
        
        lattice_type = "unknown"
        if hasattr(ipeps_state, 'unit_cell') and hasattr(ipeps_state.unit_cell, '__class__'):
            lattice_type = ipeps_state.unit_cell.__class__.__name__
        
        model_type = extra_data.get('model_type', 'unknown') if extra_data else 'unknown'
        
        metadata = CheckpointMetadata(
            version=CHECKPOINT_VERSION,
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
            energy=energy,
            convergence=convergence,
            bond_dim=bond_dim,
            lattice_type=lattice_type,
            model_type=model_type,
            description=description,
            checksum=checksum,
        )
        
        # Combine magic, metadata, and data
        full_data = {
            'magic': CHECKPOINT_MAGIC.decode('utf-8'),
            'metadata': asdict(metadata),
            'data': checkpoint_bytes,
        }
        
        # Write checkpoint
        ckpt_path = self._get_checkpoint_path(iteration)
        
        full_bytes = pickle.dumps(full_data)
        
        if self.compress:
            with gzip.open(ckpt_path, 'wb') as f:
                f.write(full_bytes)
        else:
            with open(ckpt_path, 'wb') as f:
                f.write(full_bytes)
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
        
        return ckpt_path
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = checkpoints[:-self.max_checkpoints]
            for ckpt in to_remove:
                ckpt['path'].unlink(missing_ok=True)
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        verify_checksum: bool = True,
    ) -> tuple:
        """
        Load checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file
        verify_checksum : bool
            Whether to verify data integrity
        
        Returns
        -------
        tuple
            (ipeps_state, environment, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Read checkpoint
        if checkpoint_path.suffix == '.gz':
            with gzip.open(checkpoint_path, 'rb') as f:
                full_bytes = f.read()
        else:
            with open(checkpoint_path, 'rb') as f:
                full_bytes = f.read()
        
        full_data = pickle.loads(full_bytes)
        
        # Verify magic
        if full_data.get('magic') != CHECKPOINT_MAGIC.decode('utf-8'):
            raise ValueError("Invalid checkpoint file")
        
        # Get metadata
        metadata = CheckpointMetadata(**full_data['metadata'])
        
        # Verify checksum
        if verify_checksum:
            computed_checksum = self._compute_checksum(full_data['data'])
            if computed_checksum != metadata.checksum:
                raise ValueError("Checkpoint data corrupted (checksum mismatch)")
        
        # Deserialize
        checkpoint = pickle.loads(full_data['data'])
        state, environment = self._deserialize_state(checkpoint['state'])
        
        return state, environment, metadata
    
    def load_latest(
        self,
        verify_checksum: bool = True,
    ) -> Optional[tuple]:
        """
        Load the most recent checkpoint.
        
        Returns
        -------
        tuple or None
            (ipeps_state, environment, metadata) or None if no checkpoints
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        return self.load(latest['path'], verify_checksum=verify_checksum)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Returns
        -------
        list of dict
            Checkpoint information, sorted by iteration
        """
        checkpoints = []
        
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_*.ckpt*"):
            try:
                # Read just the metadata
                if ckpt_path.suffix == '.gz':
                    with gzip.open(ckpt_path, 'rb') as f:
                        full_data = pickle.loads(f.read())
                else:
                    with open(ckpt_path, 'rb') as f:
                        full_data = pickle.loads(f.read())
                
                metadata = full_data.get('metadata', {})
                
                checkpoints.append({
                    'path': ckpt_path,
                    'iteration': metadata.get('iteration', 0),
                    'timestamp': metadata.get('timestamp'),
                    'energy': metadata.get('energy'),
                    'description': metadata.get('description'),
                })
            except Exception as e:
                warnings.warn(f"Could not read checkpoint {ckpt_path}: {e}")
        
        # Sort by iteration
        checkpoints.sort(key=lambda x: x['iteration'])
        
        return checkpoints
    
    def clean(self) -> int:
        """
        Remove all checkpoints.
        
        Returns
        -------
        int
            Number of checkpoints removed
        """
        count = 0
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_*.ckpt*"):
            ckpt_path.unlink()
            count += 1
        
        return count


class AutoCheckpointer:
    """
    Automatic checkpointing during optimization.
    
    Integrates with optimization loops to save checkpoints
    at regular intervals.
    
    Parameters
    ----------
    manager : CheckpointManager
        Checkpoint manager to use
    interval : int
        Checkpoint every N iterations
    on_improvement_only : bool
        Only checkpoint when energy improves
    """
    
    def __init__(
        self,
        manager: CheckpointManager,
        interval: int = 100,
        on_improvement_only: bool = False,
    ):
        self.manager = manager
        self.interval = interval
        self.on_improvement_only = on_improvement_only
        
        self._best_energy = float('inf')
        self._last_checkpoint_iter = -1
    
    def maybe_checkpoint(
        self,
        ipeps_state: Any,
        environment: Any,
        iteration: int,
        energy: float,
        **kwargs,
    ) -> Optional[Path]:
        """
        Save checkpoint if conditions are met.
        
        Returns path to checkpoint if saved, None otherwise.
        """
        should_save = False
        
        # Check interval
        if iteration - self._last_checkpoint_iter >= self.interval:
            should_save = True
        
        # Check improvement condition
        if self.on_improvement_only:
            if energy < self._best_energy:
                self._best_energy = energy
                should_save = True
            else:
                should_save = False
        
        if should_save:
            self._last_checkpoint_iter = iteration
            return self.manager.save(
                ipeps_state=ipeps_state,
                environment=environment,
                iteration=iteration,
                energy=energy,
                **kwargs,
            )
        
        return None
    
    def on_signal(
        self,
        ipeps_state: Any,
        environment: Any,
        iteration: int,
        energy: float,
        **kwargs,
    ) -> Path:
        """
        Emergency checkpoint on signal (e.g., SIGINT).
        
        Always saves regardless of interval or improvement.
        """
        return self.manager.save(
            ipeps_state=ipeps_state,
            environment=environment,
            iteration=iteration,
            energy=energy,
            description="Emergency checkpoint (signal)",
            **kwargs,
        )


def setup_signal_handlers(
    auto_checkpointer: AutoCheckpointer,
    ipeps_state: Any,
    environment: Any,
) -> None:
    """
    Set up signal handlers for graceful shutdown.
    
    Saves checkpoint on SIGINT (Ctrl+C) and SIGTERM.
    """
    import signal
    
    state = {'iteration': 0, 'energy': 0.0}
    
    def handler(signum, frame):
        print(f"\nReceived signal {signum}, saving checkpoint...")
        auto_checkpointer.on_signal(
            ipeps_state=ipeps_state,
            environment=environment,
            iteration=state['iteration'],
            energy=state['energy'],
        )
        raise SystemExit(f"Terminated by signal {signum}")
    
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    return state
