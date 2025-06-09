# hip_ctypes_wrapper.py

import ctypes
import numpy as np
import os

# 1) Locate the shared library (assume it’s named 'libcoredist.so' in the same folder)
_this_dir = os.path.dirname(__file__)
lib_path = os.path.join(_this_dir, "libcoredist.so")

if not os.path.isfile(lib_path):
    raise FileNotFoundError(f"Expected shared lib at '{lib_path}'")

# 2) Load the library
#    Use RTLD_LOCAL to avoid exporting symbols globally.
corelib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

# 3) Declare function signature:
#    void compute_core_distances(
#        const float* data, 
#        int N, 
#        int D, 
#        int k, 
#        float* core_out
#    );
#
# Argtypes:
#   - data        → POINTER(c_float)
#   - N, D, k     → c_int, c_int, c_int
#   - core_out    → POINTER(c_float)
#
# Restype: None (void)

corelib.compute_core_distances.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # data
    ctypes.c_int,                    # N
    ctypes.c_int,                    # D
    ctypes.c_int,                    # k
    ctypes.POINTER(ctypes.c_float)   # core_out
]
corelib.compute_core_distances.restype = None


def core_distances_gpu_hip_ctypes(data_np: np.ndarray, k: int) -> np.ndarray:
    """
    Call the HIP library’s compute_core_distances() via ctypes.

    Parameters
    ----------
    data_np : np.ndarray of shape (N, D), dtype=np.float32
        The input dataset. Must be contiguous in row-major order (C-contiguous).
    k : int
        The 'k-th neighbor' index (positive integer, 1 ≤ k < N).

    Returns
    -------
    core_out : np.ndarray of shape (N,), dtype=np.float32
        Each entry is sqrt(distance² to k-th nearest neighbor of that row).
    """
    if data_np.dtype != np.float32:
        raise ValueError("data_np must be a NumPy array of dtype float32")
    if not data_np.flags['C_CONTIGUOUS']:
        # Ensure data is C-contiguous
        data_np = np.ascontiguousarray(data_np, dtype=np.float32)

    N, D = data_np.shape
    if k < 1 or k >= N:
        raise ValueError(f"k must satisfy 1 ≤ k < N; got k={k}, N={N}")

    # 4) Prepare output buffer
    core_out = np.empty(N, dtype=np.float32)

    # 5) Obtain pointers
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr  = core_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 6) Call the function
    corelib.compute_core_distances(data_ptr, ctypes.c_int(N), ctypes.c_int(D), ctypes.c_int(k), out_ptr)

    return core_out
