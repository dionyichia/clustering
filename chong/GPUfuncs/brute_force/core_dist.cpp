#include "core_dist.hpp"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>


// TODO: add param for other distance metrics 
// calculates core distance matrix using euclidean distance
__device__ float compute_dist(const float* tileBuf,
                                  int tileM, int tileK,
                                  float& acc)
{
  const float* tileA = tileBuf;
  const float* tileB = tileBuf + tileM*tileK;
  int localRow = threadIdx.y, localCol = threadIdx.x;
  #pragma unroll
  for(int t = 0; t < tileK; ++t) {
    float va = tileA[ localRow*tileK + t ];
    float vb = tileB[ localCol*tileK + t ];
    float diff = va - vb;
    acc += diff*diff;
  }
  return sqrtf(acc);
}

// Load the (tileM × tileK) block of A into shared memory.
//   tileA: pointer to shared-mem region for A (size tileM*tileK floats)
//   A     : global pointer to A[M×K]
//   rowBase: starting row index for this block (blockIdx.y*tileM)
//   k0    : starting column index for this tile in K-dimension
//   M,K,tileK passed from kernel arguments
__device__ void load_ATile(float*       tileA,
                           const float* A,
                           int          rowBase,
                           int          k0,
                           int          M,
                           int          K,
                           int          tileK)
{
  // Each thread.x strides across the tile’s K columns.
  for(int t = threadIdx.x; t < tileK; t += blockDim.x) {
    int r = rowBase + threadIdx.y;  // global row, how many rows of K elements to skip
    int c = k0      + t;            // global col
    // boundary guard + zero-pad
    float v = 0.0f;
    if(r < M && c < K) {
      v = A[r * K + c];
    }
    // store into shared-mem: row-major within the tile
    tileA[ threadIdx.y * tileK + t ] = v;
  }
}

__global__ void pairwise_tiled(const float* A, // rows: M, cols: K
                    float* C,
                    int M, int K, int tileK)
{
  extern __shared__ float smem[]; //size of this determined by argument for bytes fed into <<< , >>> or hipLaunchKernelIGGL
  int tileM = blockDim.y;
  float acc = 0.0f;
  int rowBase = blockIdx.y * blockDim.y;
  int colBase = blockIdx.x * blockDim.x;
  int localRow = threadIdx.y;
  int localCol = threadIdx.x;
  // double buffering
  float* buf0 = smem;                              // first buffer
  float* buf1 = smem + 2* tileM*tileK;  // second buffer
  
  // prologue: k0=0
  load_ATile(buf0,A, rowBase, 0, M, K, tileK);
  load_ATile(buf0+tileM*tileK, A, colBase, 0, M, K, tileK);
  __syncthreads();
  bool useBuf0 = true;  // indicates which buffer holds the “current” data

  // processes A and B in tiles
  for(int k0 = tileK; k0 < K; k0 += tileK) {

    // decide which buffer is computing and which buffer is fetching data
    float* currBuf = useBuf0 ? buf0 : buf1;
    float* nextBuf = useBuf0 ? buf1 : buf0;

    // 1) Prefetch the next tile into nextBuf (no sync)
    load_ATile(nextBuf,A, rowBase, k0, M, K, tileK);
    load_ATile(nextBuf+tileM*tileK, A, colBase, k0,M, K, tileK);

    // 2) Compute on the “current” buffer
    compute_dist(currBuf, tileM, tileK, acc);

    // 3) barrier: now nextBuf is fully loaded
    __syncthreads();

    // 4) flip buffers
    useBuf0 = !useBuf0;
  }
  __syncthreads();
  float* lastBuf = useBuf0 ? buf0 : buf1;
  compute_dist(lastBuf, tileM, tileK, acc);

  int r = rowBase + localRow;
  int c = colBase + localCol;
  if (r < M && c < M) {
    float dist = sqrtf(acc);
    C[r*M + c] = dist;
  }
}

std::vector<float> compute_distances_gpu(
    const std::vector<float>& hA,
    int M = 16384,
    int K = 10,
    int tileM = 32,
    int tileK = 32
) {
    // 1) Bounds and shared-mem checks
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    size_t maxSmem    = prop.sharedMemPerBlock;
    size_t bufFloats  = 2u * tileM * tileK;
    size_t shmemBytes = 2u * bufFloats * sizeof(float);
    assert(shmemBytes <= maxSmem);

    // 2) Allocate + copy input
    float *d_data, *d_dist;
    hipMalloc(&d_data, (size_t)M * K * sizeof(float));
    hipMalloc(&d_dist, (size_t)M * M * sizeof(float));
    hipMemcpy(d_data, hA.data(),
              (size_t)M * K * sizeof(float),
              hipMemcpyHostToDevice);

    // 3) Launch tiled kernel
    dim3 block(tileM, tileM),
         grid((M + tileM - 1) / tileM,
              (M + tileM - 1) / tileM);
    hipLaunchKernelGGL(
        pairwise_tiled,
        grid, block,
        shmemBytes, 0,
        d_data,    
        d_dist,    
        M, K, tileK
    );
    hipDeviceSynchronize();

    // 4) Copy back full M×M distance matrix
    std::vector<float> h_dist((size_t)M * M);
    hipMemcpy(h_dist.data(),
              d_dist,                   
              (size_t)M * M * sizeof(float),
              hipMemcpyDeviceToHost);

    // 5) Cleanup
    hipFree(d_data);
    hipFree(d_dist);

    return h_dist;
}

std::vector<float> compute_core_distances(
    const std::vector<float>& distMatrix,
    int M,
    int minPts = 5
) {
    std::vector<float> coreDist(M);
    for (int i = 0; i < M; ++i) {
        // slice out row i
        auto rowBegin = distMatrix.begin() + (size_t)i * M;
        std::vector<float> row(rowBegin, rowBegin + M);

        // ignore self-distance
        row[i] = std::numeric_limits<float>::infinity();

        // find the minPts-th smallest
        std::nth_element(
            row.begin(),
            row.begin() + (minPts - 1),
            row.end()
        );
        coreDist[i] = row[minPts - 1];
    }
    return coreDist;
}
