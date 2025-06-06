// core_distances_hip.cpp
// Compile with: hipcc core_distances_hip.cpp -o core_distances_hip

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// Kernel: compute squared Euclidean distance between all pairs
__global__ void pairwise_distances(
    const float* __restrict__ data,
    float* __restrict__ dist_sq,
    int N,
    int D
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float sum = 0.0f;
        // each thread computes one entry (i,j)
        for (int d = 0; d < D; ++d) {
            float diff = data[i * D + d] - data[j * D + d];
            sum += diff * diff;
        }
        dist_sq[i * N + j] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " input_data.bin N D k output_core_dists.bin\n";
        return 1;
    }

    // Parse arguments
    const char*  input_file   = argv[1];
    int          N            = std::stoi(argv[2]);
    int          D            = std::stoi(argv[3]);
    int          k            = std::stoi(argv[4]);
    const char*  output_file  = argv[5];

    // 1) Read input_data.bin (N * D floats, row-major)
    std::vector<float> h_data;
    h_data.resize(size_t(N) * size_t(D));
    {
        std::ifstream fin(input_file, std::ios::binary);
        if (!fin) {
            std::cerr << "Error: cannot open input file " << input_file << "\n";
            return 1;
        }
        fin.read(reinterpret_cast<char*>(h_data.data()), size_t(N) * size_t(D) * sizeof(float));
        fin.close();
    }

    // 2) Allocate device buffers
    float* d_data     = nullptr;
    float* d_dist_sq  = nullptr;
    hipError_t err;

    err = hipMalloc(&d_data,    size_t(N) * size_t(D) * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_data failed\n";
        return 1;
    }
    err = hipMalloc(&d_dist_sq, size_t(N) * size_t(N) * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_dist_sq failed\n";
        return 1;
    }

    // 3) Copy data up
    err = hipMemcpy(d_data, h_data.data(), size_t(N) * size_t(D) * sizeof(float),
                    hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy to device failed\n";
        return 1;
    }

    // 4) Launch the kernel: blocks of 16×16 threads
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    hipLaunchKernelGGL(
        pairwise_distances,
        grid,
        block,
        0,        // shared mem
        0,        // stream
        d_data,
        d_dist_sq,
        N,
        D
    );
    hipDeviceSynchronize();

    // 5) Copy the complete N×N squared-distance matrix back to host
    std::vector<float> h_dist_sq;
    h_dist_sq.resize(size_t(N) * size_t(N));
    err = hipMemcpy(
        h_dist_sq.data(),
        d_dist_sq,
        size_t(N) * size_t(N) * sizeof(float),
        hipMemcpyDeviceToHost
    );
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy back failed\n";
        return 1;
    }

    // 6) Free device memory
    hipFree(d_data);
    hipFree(d_dist_sq);

    // 7) On host, compute each row's k-th neighbor distance
    //    (set diagonal to +∞, then use std::nth_element)
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> h_core_dists;
    h_core_dists.resize(N);

    for (int i = 0; i < N; ++i) {
        float* row_ptr = h_dist_sq.data() + size_t(i) * size_t(N);
        // Temporarily set self-distance to INF
        float orig_ii = row_ptr[i];
        row_ptr[i] = INF;

        // Copy this row into a small vector for selection
        std::vector<float> row(row_ptr, row_ptr + N);
        // 1-indexed semantics: to get k-th smallest, use (k-1) index
        std::nth_element(row.begin(), row.begin() + (k - 1), row.end());
        float kth_sq = row[size_t(k - 1)];
        h_core_dists[i] = std::sqrt(kth_sq);

        // (optional) restore the original diagonal
        row_ptr[i] = orig_ii;
    }

    // 8) Write out the N core distances (float32) to output_file
    {
        std::ofstream fout(output_file, std::ios::binary);
        if (!fout) {
            std::cerr << "Error: cannot open output file " << output_file << "\n";
            return 1;
        }
        fout.write(reinterpret_cast<const char*>(h_core_dists.data()),
                   size_t(N) * sizeof(float));
        fout.close();
    }

    return 0;
}
