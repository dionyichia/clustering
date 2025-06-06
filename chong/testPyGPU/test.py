# benchmark_core_distances.py

import time
import numpy as np
from sklearn.neighbors import KDTree
import subprocess
import os
import struct

def generate_data(N, D, seed=42):
    """Generate random float32 data in [0,1)^D."""
    rng = np.random.default_rng(seed)
    data = rng.random((N, D), dtype=np.float32)
    return data

def core_distances_cpu_kdtree(data: np.ndarray, k: int):
    """
    Build a KDTree and return each point's k-th nearest neighbor distance.
    We query k+1 because the 0-th neighbor is the point itself (distance=0).
    """
    tree = KDTree(data, leaf_size=40, metric='euclidean')
    dist, idx = tree.query(data, k=k+1, return_distance=True)  # shape (N, k+1)
    core_dists = dist[:, k]  # shape (N,)
    return core_dists

def core_distances_gpu_hip(data_np: np.ndarray, k: int,
                           hip_executable: str = "./core_distances_hip"):
    """
    Write `data_np` (shape N×D, float32) to a binary file, invoke the HIP executable,
    and read back an output binary of length N (float32) containing each point's
    core distance. Returns a NumPy array of length N.
    """
    N, D = data_np.shape

    # 1) Create temporary filenames:
    #    - input: "hip_input_{pid}.bin"
    #    - output: "hip_output_{pid}.bin"
    pid = os.getpid()
    input_fn  = f"hip_input_{pid}.bin"
    output_fn = f"hip_output_{pid}.bin"

    try:
        # 2) Write data_np to input_fn as raw float32 (row-major)
        with open(input_fn, "wb") as f:
            f.write(data_np.tobytes())

        # 3) Call the HIP executable:
        #    core_distances_hip <input_fn> <N> <D> <k> <output_fn>
        cmd = [
            hip_executable,
            input_fn,
            str(N),
            str(D),
            str(k),
            output_fn
        ]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"HIP executable failed:\n{stderr_txt}")

        # 4) Read back output_fn: it should contain N float32 values
        core_dists = np.empty(N, dtype=np.float32)
        with open(output_fn, "rb") as f:
            data = f.read()
            expected_bytes = N * 4
            if len(data) != expected_bytes:
                raise RuntimeError(
                    f"Expected {expected_bytes} bytes in '{output_fn}', got {len(data)} bytes."
                )
            # Interpret raw bytes as float32
            core_dists = np.frombuffer(data, dtype=np.float32)

        return core_dists

    finally:
        # 5) Clean up temporary files
        if os.path.exists(input_fn):
            os.remove(input_fn)
        if os.path.exists(output_fn):
            os.remove(output_fn)

def benchmark(N=50_000, D=10, k=5, repeat=3, hip_executable:str="./core_distances_hip"):
    """
    Run benchmarks on a random dataset of size (N, D):
    - CPU (KDTree) core-distance calculation
    - GPU (HIP) core-distance calculation (via external HIP executable)
    Returns a dict with average times and raw per-run values.
    """
    data = generate_data(N, D)

    # Warm up CPU
    print(f"Warm-up CPU KDTree (N={N}, D={D}, k={k}) ...")
    _ = core_distances_cpu_kdtree(data, k)

    # Warm up HIP ↑
    if os.path.isfile(hip_executable) and os.access(hip_executable, os.X_OK):
        print("Warm-up GPU (HIP) distance computation ...")
        _ = core_distances_gpu_hip(data, k, hip_executable=hip_executable)
    else:
        print(f"Warning: HIP executable '{hip_executable}' not found or not executable. "
              "Skipping GPU warm-up.")

    # 1) Benchmark CPU KDTree
    cpu_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cd_cpu = core_distances_cpu_kdtree(data, k)
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
    cpu_avg = float(np.mean(cpu_times))

    # 2) Benchmark GPU (HIP) if available
    gpu_avg = None
    gpu_times = None
    if os.path.isfile(hip_executable) and os.access(hip_executable, os.X_OK):
        gpu_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            cd_gpu = core_distances_gpu_hip(data, k, hip_executable=hip_executable)
            t1 = time.perf_counter()
            gpu_times.append(t1 - t0)
        gpu_avg = float(np.mean(gpu_times))

        # Sanity-check: compare CPU vs. HIP‐GPU results
        max_abs_diff = np.max(np.abs(cd_cpu - cd_gpu))
        print(f"Max abs difference between CPU / HIP results: {max_abs_diff:.5e}")
    else:
        print(f"Skipping GPU (HIP) benchmark: '{hip_executable}' not found or not executable.")

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark CPU KDTree vs. HIP‐GPU pairwise for core distances."
    )
    parser.add_argument("--N", type=int, default=30_000, help="Number of points")
    parser.add_argument("--D", type=int, default=5, help="Number of dimensions")
    parser.add_argument("--k", type=int, default=5, help="k-th nearest neighbor")
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetitions")
    parser.add_argument(
        "--hip_exec",
        type=str,
        default="./core_distances_hip",
        help="Path to the compiled HIP executable"
    )

    args = parser.parse_args()
    results = benchmark(
        N=args.N,
        D=args.D,
        k=args.k,
        repeat=args.repeat,
        hip_executable=args.hip_exec
    )
    print("\n=== Benchmark Results ===")
    print(f"N={results['N']}, D={results['D']}, k={results['k']}")
    print(f"CPU (KDTree) average time over {args.repeat} runs: {results['cpu_avg_sec']:.4f} s")
    if results['gpu_avg_sec'] is not None:
        print(f"GPU (HIP) average time over {args.repeat} runs: {results['gpu_avg_sec']:.4f} s")
    else:
        print("GPU (HIP): not available or skipped.")
