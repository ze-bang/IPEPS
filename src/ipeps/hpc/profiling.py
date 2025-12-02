"""
Profiling and performance monitoring for iPEPS computations.

Provides:
- Hierarchical timing
- Memory tracking
- FLOP counting
- Performance reports
"""

from __future__ import annotations

import time
import functools
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import warnings

try:
    import cupy as cp
    from cupy import cuda
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class TimingStats:
    """Statistics for a timed region."""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    @property
    def avg_time(self) -> float:
        """Average time per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count
    
    def add_sample(self, dt: float) -> None:
        """Add a timing sample."""
        self.total_time += dt
        self.call_count += 1
        self.min_time = min(self.min_time, dt)
        self.max_time = max(self.max_time, dt)


@dataclass  
class MemoryStats:
    """Memory usage statistics."""
    peak_cpu_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    current_cpu_mb: float = 0.0
    current_gpu_mb: float = 0.0


class Timer:
    """
    Hierarchical timer for profiling.
    
    Supports nested timing regions and GPU synchronization.
    
    Examples
    --------
    >>> timer = Timer()
    >>> with timer.region("ctmrg"):
    ...     with timer.region("absorption"):
    ...         # do absorption
    ...     with timer.region("renormalization"):
    ...         # do renormalization
    >>> timer.report()
    """
    
    def __init__(self, sync_gpu: bool = True):
        """
        Initialize timer.
        
        Parameters
        ----------
        sync_gpu : bool
            Synchronize GPU before timing (for accurate GPU timing)
        """
        self.sync_gpu = sync_gpu and HAS_CUPY
        self.stats: Dict[str, TimingStats] = {}
        self._stack: List[tuple] = []  # (name, start_time)
        self.enabled = True
    
    def _sync(self) -> None:
        """Synchronize GPU if available."""
        if self.sync_gpu:
            try:
                cuda.Device().synchronize()
            except:
                pass
    
    @contextmanager
    def region(self, name: str):
        """
        Time a code region.
        
        Parameters
        ----------
        name : str
            Name of the region
        """
        if not self.enabled:
            yield
            return
        
        # Build hierarchical name
        if self._stack:
            full_name = f"{self._stack[-1][0]}/{name}"
        else:
            full_name = name
        
        # Initialize stats if needed
        if full_name not in self.stats:
            self.stats[full_name] = TimingStats(name=full_name)
        
        # Start timing
        self._sync()
        start = time.perf_counter()
        self._stack.append((full_name, start))
        
        try:
            yield
        finally:
            # Stop timing
            self._sync()
            end = time.perf_counter()
            self._stack.pop()
            
            dt = end - start
            self.stats[full_name].add_sample(dt)
    
    def time_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to time a function.
        
        Parameters
        ----------
        name : str, optional
            Region name (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            region_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.region(region_name):
                    return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def reset(self) -> None:
        """Reset all timing statistics."""
        self.stats.clear()
        self._stack.clear()
    
    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get statistics for a region."""
        return self.stats.get(name)
    
    def report(self, sort_by: str = "total") -> str:
        """
        Generate timing report.
        
        Parameters
        ----------
        sort_by : str
            Sort by "total", "avg", "calls", or "name"
        
        Returns
        -------
        str
            Formatted report
        """
        if not self.stats:
            return "No timing data collected."
        
        # Sort
        items = list(self.stats.values())
        if sort_by == "total":
            items.sort(key=lambda x: x.total_time, reverse=True)
        elif sort_by == "avg":
            items.sort(key=lambda x: x.avg_time, reverse=True)
        elif sort_by == "calls":
            items.sort(key=lambda x: x.call_count, reverse=True)
        else:
            items.sort(key=lambda x: x.name)
        
        # Format
        lines = [
            "=" * 80,
            "TIMING REPORT",
            "=" * 80,
            f"{'Region':<40} {'Total (s)':>10} {'Avg (ms)':>10} {'Calls':>8} {'%':>6}",
            "-" * 80,
        ]
        
        total = sum(s.total_time for s in items if '/' not in s.name)
        
        for s in items:
            pct = 100 * s.total_time / total if total > 0 else 0
            lines.append(
                f"{s.name:<40} {s.total_time:>10.3f} {s.avg_time*1000:>10.2f} {s.call_count:>8} {pct:>5.1f}%"
            )
        
        lines.extend([
            "-" * 80,
            f"{'Total':<40} {total:>10.3f}",
            "=" * 80,
        ])
        
        return "\n".join(lines)


class FLOPCounter:
    """
    Count floating-point operations.
    
    Estimates FLOPs for tensor operations based on shapes.
    """
    
    def __init__(self):
        self.total_flops = 0
        self.operation_flops: Dict[str, int] = defaultdict(int)
    
    def reset(self) -> None:
        """Reset FLOP count."""
        self.total_flops = 0
        self.operation_flops.clear()
    
    def add_matmul(
        self,
        m: int,
        n: int,
        k: int,
        operation: str = "matmul",
    ) -> int:
        """
        Add FLOPs for matrix multiplication (m, k) @ (k, n).
        
        Complex multiplication: 6 real ops
        Complex addition: 2 real ops
        Total per element: 8k - 2
        """
        flops = m * n * (8 * k - 2)  # Complex ops
        self.total_flops += flops
        self.operation_flops[operation] += flops
        return flops
    
    def add_tensordot(
        self,
        shape_a: tuple,
        shape_b: tuple,
        axes: Any,
        operation: str = "tensordot",
    ) -> int:
        """Add FLOPs for tensor contraction."""
        # Determine contracted and output dimensions
        if isinstance(axes, int):
            axes_a = list(range(-axes, 0))
            axes_b = list(range(axes))
        else:
            axes_a, axes_b = axes
        
        # Contracted dimension
        k = 1
        for ax in axes_a:
            k *= shape_a[ax]
        
        # Output dimensions
        m = 1
        for i, d in enumerate(shape_a):
            if i not in axes_a and i - len(shape_a) not in axes_a:
                m *= d
        
        n = 1
        for i, d in enumerate(shape_b):
            if i not in axes_b and i - len(shape_b) not in axes_b:
                n *= d
        
        return self.add_matmul(m, n, k, operation)
    
    def add_svd(
        self,
        m: int,
        n: int,
        operation: str = "svd",
    ) -> int:
        """Add FLOPs for SVD (estimate)."""
        # SVD is O(mn * min(m,n)) for the main algorithm
        k = min(m, n)
        flops = 4 * m * n * k + 8 * k * k * k
        self.total_flops += flops
        self.operation_flops[operation] += flops
        return flops
    
    def add_qr(
        self,
        m: int,
        n: int,
        operation: str = "qr",
    ) -> int:
        """Add FLOPs for QR decomposition."""
        k = min(m, n)
        flops = 2 * m * n * k - (2 * n * k * k) // 3
        self.total_flops += flops
        self.operation_flops[operation] += flops
        return flops
    
    def report(self) -> str:
        """Generate FLOP count report."""
        lines = [
            "=" * 60,
            "FLOP REPORT",
            "=" * 60,
        ]
        
        total = self.total_flops
        
        for op, flops in sorted(
            self.operation_flops.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = 100 * flops / total if total > 0 else 0
            lines.append(
                f"{op:<30} {flops/1e9:>12.2f} GFLOPs ({pct:>5.1f}%)"
            )
        
        lines.extend([
            "-" * 60,
            f"{'Total':<30} {total/1e9:>12.2f} GFLOPs",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class MemoryTracker:
    """
    Track memory usage during computation.
    
    Monitors CPU and GPU memory.
    """
    
    def __init__(self):
        self.samples: List[Dict[str, float]] = []
        self.peak_cpu = 0.0
        self.peak_gpu = 0.0
    
    def _get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not HAS_CUPY:
            return 0.0
        
        try:
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / (1024 * 1024)
        except:
            return 0.0
    
    def sample(self, label: str = "") -> Dict[str, float]:
        """
        Take a memory sample.
        
        Parameters
        ----------
        label : str
            Label for this sample
        
        Returns
        -------
        dict
            Memory usage
        """
        cpu = self._get_cpu_memory()
        gpu = self._get_gpu_memory()
        
        self.peak_cpu = max(self.peak_cpu, cpu)
        self.peak_gpu = max(self.peak_gpu, gpu)
        
        sample = {
            'label': label,
            'timestamp': time.time(),
            'cpu_mb': cpu,
            'gpu_mb': gpu,
        }
        
        self.samples.append(sample)
        return sample
    
    def reset(self) -> None:
        """Reset memory tracking."""
        self.samples.clear()
        self.peak_cpu = 0.0
        self.peak_gpu = 0.0
    
    @contextmanager
    def track(self, label: str):
        """
        Track memory for a code region.
        
        Parameters
        ----------
        label : str
            Label for this region
        """
        self.sample(f"{label}:start")
        try:
            yield
        finally:
            self.sample(f"{label}:end")
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        current_cpu = self._get_cpu_memory()
        current_gpu = self._get_gpu_memory()
        
        return MemoryStats(
            peak_cpu_mb=self.peak_cpu,
            peak_gpu_mb=self.peak_gpu,
            current_cpu_mb=current_cpu,
            current_gpu_mb=current_gpu,
        )
    
    def report(self) -> str:
        """Generate memory report."""
        stats = self.get_stats()
        
        lines = [
            "=" * 50,
            "MEMORY REPORT",
            "=" * 50,
            f"Peak CPU:     {stats.peak_cpu_mb:>10.1f} MB",
            f"Peak GPU:     {stats.peak_gpu_mb:>10.1f} MB",
            f"Current CPU:  {stats.current_cpu_mb:>10.1f} MB",
            f"Current GPU:  {stats.current_gpu_mb:>10.1f} MB",
            "=" * 50,
        ]
        
        return "\n".join(lines)


class Profiler:
    """
    Comprehensive profiler combining timing, FLOPs, and memory.
    
    Examples
    --------
    >>> profiler = Profiler()
    >>> with profiler.profile("ctmrg"):
    ...     # CTMRG computation
    >>> print(profiler.report())
    """
    
    def __init__(self, sync_gpu: bool = True):
        self.timer = Timer(sync_gpu=sync_gpu)
        self.flops = FLOPCounter()
        self.memory = MemoryTracker()
        self.enabled = True
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code region."""
        if not self.enabled:
            yield
            return
        
        self.memory.sample(f"{name}:start")
        
        with self.timer.region(name):
            yield
        
        self.memory.sample(f"{name}:end")
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.timer.reset()
        self.flops.reset()
        self.memory.reset()
    
    def report(self) -> str:
        """Generate comprehensive profiling report."""
        lines = [
            self.timer.report(),
            "",
            self.flops.report(),
            "",
            self.memory.report(),
        ]
        
        # Performance summary
        stats = self.timer.stats
        flop_count = self.flops.total_flops
        
        total_time = sum(s.total_time for s in stats.values() if '/' not in s.name)
        
        if total_time > 0 and flop_count > 0:
            gflops_per_sec = (flop_count / 1e9) / total_time
            lines.extend([
                "",
                "=" * 50,
                "PERFORMANCE SUMMARY",
                "=" * 50,
                f"Sustained: {gflops_per_sec:.2f} GFLOP/s",
                "=" * 50,
            ])
        
        return "\n".join(lines)


# Global profiler instance
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def profile_region(name: str):
    """Context manager for profiling a region with global profiler."""
    return get_profiler().profile(name)


def profile_function(name: Optional[str] = None) -> Callable:
    """Decorator for profiling a function with global profiler."""
    return get_profiler().timer.time_function(name)
