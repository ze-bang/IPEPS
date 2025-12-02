#!/usr/bin/env python
"""
Benchmarking iPEPS performance.

This script benchmarks the performance of key iPEPS operations:
- Tensor contractions
- CTMRG iteration
- Simple Update step
- Full Update step

Supports CPU and GPU backends.

Usage:
    python benchmark.py [--backend numpy|cupy|pytorch] [--D BOND_DIMS] 
                        [--chi ENV_DIMS] [--n_iter N_ITERATIONS]
"""

import numpy as np
import argparse
import time
from pathlib import Path

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipeps.core.tensor import Tensor, set_backend, get_backend
from ipeps.core.contractions import contract
from ipeps.hpc.profiling import Timer, MemoryTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="iPEPS performance benchmark")
    
    parser.add_argument(
        "--backend", type=str, default="numpy",
        choices=["numpy", "cupy", "pytorch"],
        help="Compute backend (default: numpy)"
    )
    parser.add_argument(
        "--D", type=int, nargs="+", default=[2, 3, 4, 5, 6],
        help="Bond dimensions to test (default: 2 3 4 5 6)"
    )
    parser.add_argument(
        "--chi", type=int, nargs="+", default=[10, 20, 40, 60],
        help="Environment dimensions to test (default: 10 20 40 60)"
    )
    parser.add_argument(
        "--d", type=int, default=2,
        help="Physical dimension (default: 2)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=10,
        help="Number of iterations for averaging (default: 10)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations (default: 2)"
    )
    
    return parser.parse_args()


def benchmark_contraction(D: int, d: int, n_iter: int, warmup: int):
    """
    Benchmark tensor contraction (core PEPS operation).
    
    Tests contraction of two 5-leg tensors along one leg.
    """
    timer = Timer()
    
    # Create random tensors
    A = Tensor.random((d, D, D, D, D), dtype=np.complex128)
    B = Tensor.random((d, D, D, D, D), dtype=np.complex128)
    
    # Warmup
    for _ in range(warmup):
        C = A.contract(B, ([3], [1]))  # Contract right-left
    
    # Benchmark
    with timer.region("contraction"):
        for _ in range(n_iter):
            C = A.contract(B, ([3], [1]))
    
    stats = timer.get_stats("contraction")
    return stats.avg_time * 1000  # ms


def benchmark_double_layer(D: int, d: int, n_iter: int, warmup: int):
    """
    Benchmark double-layer tensor construction.
    
    This is the main operation in CTMRG: contracting A with A*.
    """
    timer = Timer()
    
    # Create iPEPS tensor: (phys, left, up, right, down)
    A = Tensor.random((d, D, D, D, D), dtype=np.complex128)
    A_conj = A.conj()
    
    # Warmup
    for _ in range(warmup):
        # Contract physical indices: A_{pulrd} * A*_{p'ulrd}
        AA = contract(
            "pulrd,qulrd->ulrdulrd",
            A.data, A_conj.data
        )
    
    # Benchmark
    with timer.region("double_layer"):
        for _ in range(n_iter):
            AA = contract(
                "pulrd,qulrd->ulrdulrd",
                A.data, A_conj.data
            )
    
    stats = timer.get_stats("double_layer")
    return stats.avg_time * 1000  # ms


def benchmark_ctmrg_move(D: int, chi: int, n_iter: int, warmup: int):
    """
    Benchmark single CTMRG absorption move.
    
    This involves contracting corner-edge-double_layer-edge-corner.
    """
    timer = Timer()
    
    # Create environment tensors
    C1 = Tensor.random((chi, chi), dtype=np.complex128)  # Corner
    T1 = Tensor.random((chi, D*D, chi), dtype=np.complex128)  # Edge
    a = Tensor.random((D*D, D*D, D*D, D*D), dtype=np.complex128)  # Double layer
    
    # Warmup
    for _ in range(warmup):
        # C1 - T1 - a - T1 - C1 type contraction
        temp = contract("ij,jkl->ikl", C1.data, T1.data)
        temp = contract("ikl,kmno->ilmno", temp, a.data)
    
    # Benchmark
    with timer.region("ctmrg_move"):
        for _ in range(n_iter):
            temp = contract("ij,jkl->ikl", C1.data, T1.data)
            temp = contract("ikl,kmno->ilmno", temp, a.data)
    
    stats = timer.get_stats("ctmrg_move")
    return stats.avg_time * 1000  # ms


def benchmark_svd(D: int, chi: int, n_iter: int, warmup: int):
    """
    Benchmark SVD (used in bond truncation).
    """
    timer = Timer()
    
    # Matrix shape typical for CTMRG: (chi * D^2) x (chi * D^2)
    m = chi * D * D
    M = Tensor.random((m, m), dtype=np.complex128)
    
    # Warmup
    for _ in range(warmup):
        U, S, Vh = M.svd()
    
    # Benchmark
    with timer.region("svd"):
        for _ in range(n_iter):
            U, S, Vh = M.svd()
    
    stats = timer.get_stats("svd")
    return stats.avg_time * 1000  # ms


def benchmark_simple_update(D: int, d: int, n_iter: int, warmup: int):
    """
    Benchmark Simple Update bond update step.
    """
    timer = Timer()
    
    # Create tensors and gate
    A = Tensor.random((d, D, D, D, D), dtype=np.complex128)
    B = Tensor.random((d, D, D, D, D), dtype=np.complex128)
    lam = Tensor.random((D,), dtype=np.float64)  # Singular values
    gate = Tensor.random((d, d, d, d), dtype=np.complex128)  # Two-site gate
    
    # Warmup
    for _ in range(warmup):
        # Contract A-lambda-B
        A_lam = contract("pulrd,r->pulrd", A.data, np.sqrt(np.abs(lam.data)))
        B_lam = contract("pulrd,l->pulrd", B.data, np.sqrt(np.abs(lam.data)))
        
        # Apply gate
        theta = contract("pulrd,qvlre->pqulde", A_lam, B_lam)
        theta = contract("pqulde,pqrs->rsulde", theta, gate.data)
        
        # SVD truncation
        shape = theta.shape
        theta_mat = theta.reshape(d * D * D, d * D * D)
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
    
    # Benchmark
    with timer.region("simple_update"):
        for _ in range(n_iter):
            A_lam = contract("pulrd,r->pulrd", A.data, np.sqrt(np.abs(lam.data)))
            B_lam = contract("pulrd,l->pulrd", B.data, np.sqrt(np.abs(lam.data)))
            theta = contract("pulrd,qvlre->pqulde", A_lam, B_lam)
            theta = contract("pqulde,pqrs->rsulde", theta, gate.data)
            theta_mat = theta.reshape(d * D * D, d * D * D)
            U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
    
    stats = timer.get_stats("simple_update")
    return stats.avg_time * 1000  # ms


def run_benchmarks(args):
    """Run all benchmarks."""
    print("=" * 70)
    print(f"iPEPS Performance Benchmark - Backend: {args.backend}")
    print("=" * 70)
    
    # Set backend
    try:
        set_backend(args.backend)
    except ValueError as e:
        print(f"Warning: {e}, falling back to numpy")
        set_backend("numpy")
    
    print(f"Active backend: {get_backend()}")
    print()
    
    # Track memory
    memory = MemoryTracker()
    memory.sample("start")
    
    results = {}
    
    # Contraction benchmark
    print("Tensor Contraction (two 5-leg tensors):")
    print("-" * 50)
    print(f"{'D':>6} {'Time (ms)':>12} {'GFLOP/s':>12}")
    print("-" * 50)
    
    contraction_times = []
    for D in args.D:
        t = benchmark_contraction(D, args.d, args.n_iter, args.warmup)
        contraction_times.append(t)
        
        # Estimate FLOPs: O(d * D^9) for this contraction
        flops = args.d * D**9 * 8  # Complex multiply-add
        gflops = flops / (t * 1e-3) / 1e9
        
        print(f"{D:>6} {t:>12.3f} {gflops:>12.2f}")
    
    results['contraction'] = {'D': args.D, 'time_ms': contraction_times}
    print()
    
    # Double layer benchmark
    print("Double Layer Construction:")
    print("-" * 50)
    print(f"{'D':>6} {'Time (ms)':>12}")
    print("-" * 50)
    
    double_layer_times = []
    for D in args.D:
        t = benchmark_double_layer(D, args.d, args.n_iter, args.warmup)
        double_layer_times.append(t)
        print(f"{D:>6} {t:>12.3f}")
    
    results['double_layer'] = {'D': args.D, 'time_ms': double_layer_times}
    print()
    
    # CTMRG move benchmark
    print("CTMRG Absorption Move:")
    print("-" * 50)
    print(f"{'D':>6} {'chi':>6} {'Time (ms)':>12}")
    print("-" * 50)
    
    ctmrg_results = []
    for D in args.D[:3]:  # Limit D for CTMRG
        for chi in args.chi:
            t = benchmark_ctmrg_move(D, chi, args.n_iter, args.warmup)
            ctmrg_results.append({'D': D, 'chi': chi, 'time_ms': t})
            print(f"{D:>6} {chi:>6} {t:>12.3f}")
    
    results['ctmrg'] = ctmrg_results
    print()
    
    # SVD benchmark
    print("SVD (chi*D^2 x chi*D^2 matrix):")
    print("-" * 50)
    print(f"{'D':>6} {'chi':>6} {'Size':>10} {'Time (ms)':>12}")
    print("-" * 50)
    
    svd_results = []
    for D in args.D[:3]:
        for chi in args.chi[:3]:
            size = chi * D * D
            t = benchmark_svd(D, chi, args.n_iter, args.warmup)
            svd_results.append({'D': D, 'chi': chi, 'size': size, 'time_ms': t})
            print(f"{D:>6} {chi:>6} {size:>10} {t:>12.3f}")
    
    results['svd'] = svd_results
    print()
    
    # Simple Update benchmark
    print("Simple Update Bond Step:")
    print("-" * 50)
    print(f"{'D':>6} {'Time (ms)':>12}")
    print("-" * 50)
    
    su_times = []
    for D in args.D:
        t = benchmark_simple_update(D, args.d, args.n_iter, args.warmup)
        su_times.append(t)
        print(f"{D:>6} {t:>12.3f}")
    
    results['simple_update'] = {'D': args.D, 'time_ms': su_times}
    print()
    
    # Memory report
    memory.sample("end")
    print(memory.report())
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    results = run_benchmarks(args)
    
    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
