#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
// This is supposed to perform tiling to reduce the global memory accesses 
// This is done by loading subsections of the data into local memory, performing relevant calculations
// Rather than each thread loading the same data repeatedly
// total 64Kb, allocated across blocks
// in each blocks, assigned to each thread respectively

//-------------------------------------------------------------
// Naive M×M distance calculator (Euclidean).
// A:  pointer to M×K data, row-major
// B:  pointer to N×K data, row-major  (for core-distances, pass B==A and N==M)
// C:  pointer to M×N output distances
// M:  number of “rows”  (points in A)
// N:  number of “cols”  (points in B)
// K:  feature dimension
//-------------------------------------------------------------
__global__ void pairwise_naive(
    const float* A,
    const float* B,
    float*       C,
    int          M,
    int          N,
    int          K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    // accumulate squared-diffs
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
      float va = A[row*K + k];
      float vb = B[col*K + k];
      float d  = va - vb;
      acc += d*d;
    }
    // write the *actual* Euclidean distance
    C[row*N + col] = sqrtf(acc);
  }
}

// TODO: add param for other distance metrics 
// calculates core distance matrix using euclidean distance  
__device__ void compute_core_dist(const float* tileBuf,
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
                    const float* B, // rows: K, cols: N
                    float* C,
                    int M, int N, int K, int tileK)
{
  extern __shared__ float smem[]; //size of this determined by argument for bytes fed into <<< , >>> or hipLaunchKernelIGGL
  // tileM and tileN come from blockDim at runtime:
  int tileN = blockDim.x;
  int tileM = blockDim.y;
  float acc = 0.0f;

  int rowBase = blockIdx.y * blockDim.y;
  int colBase = blockIdx.x * blockDim.x;
  int localRow = threadIdx.y;
  int localCol = threadIdx.x;
  // double buffering
  float* buf0 = smem;                              // first buffer
  float* buf1 = smem + tileM*tileK + tileK*tileN;  // second buffer

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
    compute_core_dist(currBuf, tileM, tileK, acc);

    // 3) barrier: now nextBuf is fully loaded
    __syncthreads();

    // 4) flip buffers
    useBuf0 = !useBuf0;
  }
  __syncthreads();
  float* lastBuf = useBuf0 ? buf0 : buf1;
  compute_core_dist(lastBuf, tileM, tileK, acc);

  int r = rowBase + localRow;
  int c = colBase + localCol;
  if (r < M && c < N) {
    float dist = sqrtf(acc);
    C[r*N + c] = dist;
  }

}

int main(int argc, char** argv) {

  // Defaults
  // dimensions of tiles used in tiling Blocks
  int tileM = 32, tileK = 32;
  // minPts input to HDBSCAN 
  int minPts = 5;
  // M == rows of input A, number of datapoints
  // K == columns of input A, number of features in one datapoint
  int M = 1024, K = 10;

  // Parse param overrides (if any)
  int i = 1;
  while (i + 1 < argc) {
      if (!strcmp(argv[i], "--tile-m")) {
        if(argv[i+1] <= 32){
          tileM  = std::atoi(argv[i+1]);
          i += 2;               // consumed flag + value
        }
        else{
          std::cout<<"Max Tile Dimension is 32"<<"\n";
          return;
        }
      }
      else if (!strcmp(argv[i], "--tile-k")) {
        if(argv[i+1] <= 32){
          tileM  = std::atoi(argv[i+1]);
          i += 2;               // consumed flag + value
        }
        else{
          std::cout<<"Max Tile Dimension is 32"<<"\n";
          return;
        }
      }
      else if (!strcmp(argv[i], "--minpts")) {
          minPts = std::atoi(argv[i+1]);
          i += 2;
      }
      else if (!strcmp(argv[i], "--M")) {
          M      = std::atoi(argv[i+1]);
          i += 2;
      }
      else if (!strcmp(argv[i], "--K")) {
          K      = std::atoi(argv[i+1]);
          i += 2;
      }
      else {
          // unrecognized flag: skip just the flag
          std::cerr << "Warning: unknown option '" << argv[i] << "'\n";
          i += 1;
      }
  }

  // Ensures tile size does not exceed dimensions of input 
  tileM = std::min(tileM, M);
  tileK = std::min(tileK, K);

  // Check shared memory budget,
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  size_t maxSmem = prop.sharedMemPerBlock;
  size_t bufFloats = size_t(tileM)*tileK + size_t(tileM)*tileK;
  size_t shmemTiled = 2 * bufFloats * sizeof(float);
  assert((bufFloats * sizeof(float)) <= maxSmem);
  assert((shmemTiled) <= maxSmem);

  std::cout<<"Problem: M="<<M<<" K="<<K<<"\n"
           <<"MinPts: "<<minPts<<"\n"
           <<"Tiles: ("<<tileM<<"×"<<tileK<<" and "<<tileM<<"×"<<tileK<<")\n"
           <<"Shared mem needed (single buf): "<<(bufFloats*sizeof(float))/1024<<" KB\n"
           <<"Shared mem needed (double buf): "<<(shmemTiled)/1024<<" KB\n";

  // Allocate host data
  std::vector<float> hA(M*K);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist;
  for(auto &x: hA) x = dist(rng);

  // Allocate device data
  float *dA, *dC_naive, *dC_tiled;
  hipMalloc(&dA, M*K*sizeof(float));
  hipMalloc(&dC_naive, M*M*sizeof(float));
  hipMalloc(&dC_tiled, M*M*sizeof(float));

  // Copy inputs
  hipMemcpy(dA, hA.data(), M*K*sizeof(float), hipMemcpyHostToDevice);

  // Launch configs
  dim3 threads_naive(16,16),  // you can tune this
       grid_naive((M+15)/16,(M+15)/16);

  // number of threads per block is tileM x tileM
  // shared memory is accessed in segments of tileM x tileK
  dim3 threads_tiled(tileM, tileM),
       grid_tiled((M+tileM-1)/tileM,(M+tileM-1)/tileM);

  // Create HIP events
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  const int ITER = 50;

  // Warm-up
  hipLaunchKernelGGL(pairwise_naive,
                     grid_naive, threads_naive, 0, 0,
                     dA,dA,dC_naive,M,M,K);
  hipLaunchKernelGGL(pairwise_tiled,
                     grid_tiled, threads_tiled, shmemTiled, 0,
                     dA,dA,dC_tiled,M,M,K,tileK);
  hipDeviceSynchronize();

  // Time naive
  hipEventRecord(start);
  for(int i=0; i<ITER; ++i) {
    hipLaunchKernelGGL(pairwise_naive,
                       grid_naive, threads_naive, 0, 0,
                       dA,dA,dC_naive,M,M,K);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms_naive;
  hipEventElapsedTime(&ms_naive, start, stop);
  ms_naive /= ITER;

  // Time tiled
  hipEventRecord(start);
  for(int i=0; i<ITER; ++i) {
    hipLaunchKernelGGL(pairwise_tiled,
                      grid_tiled, threads_tiled, shmemTiled, 0,
                      dA,dA,dC_tiled,M,M,K,tileK);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms_tiled;
  hipEventElapsedTime(&ms_tiled, start, stop);
  ms_tiled /= ITER;

  // Copy back for a quick correctness check
  std::vector<float> hC_naive(M*M), hC_tiled(M*M);
  hipMemcpy(hC_naive.data(), dC_naive, M*M*sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hC_tiled.data(), dC_tiled, M*M*sizeof(float), hipMemcpyDeviceToHost);

  // CPU: Compute core distances
  std::vector<float> coreDist(M);
  std::vector<float> coreDist_naive(M);
  for(int i = 0; i < M; ++i) {
    // gather row i
    auto rowBegin = hC_tiled.begin() + i*M;
    auto rowBegin_naive = hC_naive.begin() + i*M;
    std::vector<float> row(rowBegin, rowBegin + M);
    std::vector<float> row_naive(rowBegin_naive, rowBegin_naive + M);
    // ignore self-distance
    row[i] = std::numeric_limits<float>::infinity();
    row_naive[i] = std::numeric_limits<float>::infinity();
    // nth_element to find the k-th smallest
    std::nth_element(row.begin(),
                    row.begin() + (minPts - 1),
                    row.end());
    std::nth_element(row_naive.begin(),
                    row_naive.begin() + (minPts - 1),
                    row_naive.end());
    coreDist[i] = row[minPts - 1];
    coreDist_naive[i] = row_naive[minPts - 1];
  }

  // Validate
  double maxErr = 0;
  for(int i=0;i<M;++i){
    double diff = std::fabs(static_cast<double>(coreDist_naive[i] - coreDist[i]));
    maxErr = std::max<double>(maxErr, diff);
  }
  std::cout<<"Max error between naive/tiled: "<<maxErr<<"\n";

  // Compute GFLOPS (2 FLOPs per multiply-add)
  double flops = double(M)*M*2*K;
  double g_naive = flops / (ms_naive*1e-3) / 1e9;
  double g_tiled = flops / (ms_tiled*1e-3) / 1e9;
  double speedup = ms_naive / ms_tiled;

  std::cout<<"Naive: "<<ms_naive<<" ms  ("<<g_naive<<" GFLOPS)\n"
           <<"Tiled: "<<ms_tiled<<" ms  ("<<g_tiled<<" GFLOPS)\n"
           <<"Speedup: "<<speedup<<"×\n";

  // Cleanup
  hipFree(dA);
  hipFree(dC_naive); 
  hipFree(dC_tiled);
  hipEventDestroy(start); 
  hipEventDestroy(stop);

  return 0;
}