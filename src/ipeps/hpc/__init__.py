"""
HPC infrastructure module for iPEPS.

Provides:
- MPI parallelization
- GPU acceleration
- Checkpointing/restart
- Performance profiling
"""

from ipeps.hpc.mpi import MPIManager, distribute_bonds, gather_results
from ipeps.hpc.gpu import (
    GPUBackend,
    GPUConfig,
    MultiGPUBackend,
    get_available_gpus,
    optimize_for_gpu,
)
from ipeps.hpc.checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
    AutoCheckpointer,
    setup_signal_handlers,
)
from ipeps.hpc.profiling import (
    Profiler,
    Timer,
    FLOPCounter,
    MemoryTracker,
    get_profiler,
    profile_region,
    profile_function,
)

__all__ = [
    # MPI
    "MPIManager",
    "distribute_bonds",
    "gather_results",
    # GPU
    "GPUBackend",
    "GPUConfig",
    "MultiGPUBackend",
    "get_available_gpus",
    "optimize_for_gpu",
    # Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    "AutoCheckpointer",
    "setup_signal_handlers",
    # Profiling
    "Profiler",
    "Timer",
    "FLOPCounter",
    "MemoryTracker",
    "get_profiler",
    "profile_region",
    "profile_function",
]
