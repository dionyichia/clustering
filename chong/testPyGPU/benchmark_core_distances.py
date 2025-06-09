# benchmark_core_distances_ctypes.py

import time
import numpy as np
from sklearn.neighbors import KDTree
import os
import argparse

# Import the ctypes‐based HIP wrapper
from hip_ctypes_wrapper import core_distances_gpu_hip_ctypes


def generate_data(N, D, seed=42):
    """Generate random float32 data in [0,1)^D."""
    rng = np.random.default_rng(seed)
    data = rng.random((N, D), dtype=np.float32)
    return data

def core_distances_cpu_kdtree(data: np.ndarray, k: int) -> np.ndarray:
    """
    Build a KDTree and return each point's k-th nearest neighbor distance.
    We query k+1 because the 0-th neighbor is the point itself (distance=0).
    """
    tree = KDTree(data, leaf_size=40, metric='euclidean')
    dist, idx = tree.query(data, k=k+1, return_distance=True)  # shape (N, k+1)
    core_dists = dist[:, k]  # shape (N,)
    return core_dists

def benchmark(N=50000, D=10, k=5, repeat=3):
    """
    Run benchmarks on a random dataset of size (N, D):
    - CPU (KDTree) core-distance calculation
    - GPU (HIP via ctypes) core-distance calculation

    Returns
    -------
    result : dict
      {
        'N': N,
        'D': D,
        'k': k,
        'cpu_avg_sec': float,
        'gpu_avg_sec': float,
        'cpu_times': [float,...],
        'gpu_times': [float,...]
      }
    """
    data = generate_data(N, D)

    # ----------------------------------------------
    # 1) Warm up CPU
    print(f"Warm-up CPU KDTree (N={N}, D={D}, k={k}) ...")
    _ = core_distances_cpu_kdtree(data, k)

    # 2) Warm up GPU (HIP ctypes) on a tiny subset (N=64)
    small_N = min(64, N)
    if small_N > 1:
        k_small = min(k, small_N - 1)
    else:
        k_small = 1
    print(f"Warm-up GPU (HIP-ctypes) on tiny subset (N={small_N}, k={k_small}) ...")
    _ = core_distances_gpu_hip_ctypes(data[:small_N], k_small)
    # ----------------------------------------------

    # 3) Benchmark CPU KDTree
    cpu_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cd_cpu = core_distances_cpu_kdtree(data, k)
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
    cpu_avg = float(np.mean(cpu_times))

    # 4) Benchmark GPU (HIP-ctypes)
    gpu_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cd_gpu = core_distances_gpu_hip_ctypes(data, k)
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)
    gpu_avg = float(np.mean(gpu_times))

    # 5) Sanity-check: compare CPU vs. GPU results
    max_abs_diff = np.max(np.abs(cd_cpu - cd_gpu))
    print(f"Max abs difference CPU vs. HIP-ctypes: {max_abs_diff:.5e}")

    return {
        'N': N,
        'D': D,
        'k': k,
        'cpu_avg_sec': cpu_avg,
        'gpu_avg_sec': gpu_avg,
        'cpu_times': cpu_times,
        'gpu_times': gpu_times
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark CPU KDTree vs. HIP-ctypes for core distances."
    )
    parser.add_argument("--N", type=int, default=30000, help="Number of points")
    parser.add_argument("--D", type=int, default=5, help="Number of dimensions")
    parser.add_argument("--k", type=int, default=5, help="k-th nearest neighbor")
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetitions")

    args = parser.parse_args()
    results = benchmark(
        N=args.N,
        D=args.D,
        k=args.k,
        repeat=args.repeat
    )
    print("\n=== Benchmark Results ===")
    print(f"N={results['N']}, D={results['D']}, k={results['k']}")
    print(f"CPU (KDTree) average time over {args.repeat} runs: {results['cpu_avg_sec']:.4f} s")
    print(f"GPU (HIP-ctypes) average time over {args.repeat} runs: {results['gpu_avg_sec']:.4f} s")
