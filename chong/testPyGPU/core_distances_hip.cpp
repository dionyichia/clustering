// core_distances_hip_lib.cpp

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdint>

extern "C" {

/*
 * Kernel: for each (i,j), compute squared Euclidean distance between point i and j.
 *   - data: float[N * D], row-major
 *   - dist_sq: float[N * N], row-major (output)
 */
__global__ void pairwise_distances_kernel(
    const float* __restrict__ data,
    float*       __restrict__ dist_sq,
    int          N,
    int          D
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float sum = 0.0f;
        // Compute ∥ data[i,:] – data[j,:] ∥²
        for (int d = 0; d < D; ++d) {
            float diff = data[i * D + d] - data[j * D + d];
            sum += diff * diff;
        }
        dist_sq[i * N + j] = sum;
    }
}

/*
 * C-callable function:
 *   data:     pointer to N*D floats (row-major)
 *   N, D, k:  ints
 *   core_out: pointer to N floats (pre-allocated by caller)
 *
 * Steps:
 *   1. Allocate d_data (N*D) and d_dist_sq (N*N) on GPU
 *   2. Copy data → d_data
 *   3. Launch pairwise_distances_kernel with grid/block covering N×N 
 *   4. hipMemcpy back d_dist_sq → h_dist_sq (host buffer of length N*N)
 *   5. For each i in [0..N-1]:
 *        - set h_dist_sq[i*N + i] = +∞
 *        - place that N-length row into a std::vector<float>
 *        - nth_element to find (k-1) index (0-based), take sqrt
 *        - store in core_out[i]
 *   6. Free all GPU and host temporary memory
 */
void compute_core_distances(
    const float* data,
    int          N,
    int          D,
    int          k,
    float*       core_out
) {
    // 1) Allocate device memory
    float* d_data    = nullptr;
    float* d_dist_sq = nullptr;
    hipError_t err;

    err = hipMalloc(&d_data,    size_t(N) * size_t(D) * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc d_data failed\n");
        return;
    }
    err = hipMalloc(&d_dist_sq, size_t(N) * size_t(N) * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc d_dist_sq failed\n");
        hipFree(d_data);
        return;
    }

    // 2) Copy host data → d_data
    err = hipMemcpy(
        d_data,
        data,
        size_t(N) * size_t(D) * sizeof(float),
        hipMemcpyHostToDevice
    );
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy to device failed\n");
        hipFree(d_data);
        hipFree(d_dist_sq);
        return;
    }

    // 3) Launch the kernel
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    hipLaunchKernelGGL(
        pairwise_distances_kernel,
        grid,
        block,
        0,    // shared mem
        0,    // stream
        d_data,
        d_dist_sq,
        N,
        D
    );
    hipDeviceSynchronize();

    // 4) Copy d_dist_sq → host buffer h_dist_sq
    float* h_dist_sq = (float*)std::malloc(size_t(N) * size_t(N) * sizeof(float));
    if (!h_dist_sq) {
        fprintf(stderr, "malloc h_dist_sq failed\n");
        hipFree(d_data);
        hipFree(d_dist_sq);
        return;
    }
    err = hipMemcpy(
        h_dist_sq,
        d_dist_sq,
        size_t(N) * size_t(N) * sizeof(float),
        hipMemcpyDeviceToHost
    );
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy back failed\n");
        std::free(h_dist_sq);
        hipFree(d_data);
        hipFree(d_dist_sq);
        return;
    }

    // 5) Compute core distances on host
    const float INF = std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; ++i) {
        float* row_ptr = h_dist_sq + size_t(i) * size_t(N);
        float  orig    = row_ptr[i];       // original diagonal
        row_ptr[i] = INF;                  // ignore self-distance

        // Copy this row into a vector to find k-th smallest
        std::vector<float> row(row_ptr, row_ptr + N);
        std::nth_element(row.begin(), row.begin() + (k - 1), row.end());
        float kth_sq = row[size_t(k - 1)];
        core_out[i] = std::sqrt(kth_sq);

        row_ptr[i] = orig;  // restore (optional)
    }

    // 6) Free host and device memory
    std::free(h_dist_sq);
    hipFree(d_data);
    hipFree(d_dist_sq);
}

} // extern "C"
