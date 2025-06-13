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

// basic function to test against without any tiling or double-buffering
__global__ void pairwise_naive(const float* A, const float* B, float* C,
                               int M, int N, int K) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[row*K + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
  }
}

// Perform the inner-product on one tile
__device__ void compute_tile(const float* tileBuf,
                             int tileM, int tileK, int tileN,
                             float& acc) 
{
  // tileBuf holds first tileM×tileK floats, then tileK×tileN floats
  const float* tileA = tileBuf;
  const float* tileB = tileBuf + tileM*tileK;
  #pragma unroll
  for(int t = 0; t < tileK; ++t) {
    float va = tileA[ threadIdx.y*tileK + t ];
    float vb = tileB[ t*tileN + threadIdx.x ];
    acc += va * vb;
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
    int r = rowBase + threadIdx.y;  // global row
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

// Load the (tileK × tileN) block of B into shared memory.
//   tileB: pointer to shared-mem region for B (size tileK*tileN floats)
//   B     : global pointer to B[K×N]
//   colBase: starting column index for this block (blockIdx.x*tileN)
//   k0    : starting row index for this tile in K-dimension
//   K,N,tileK passed from kernel arguments
__device__ void load_BTile(float*       tileB,
                           const float* B,
                           int          colBase,
                           int          k0,
                           int          K,
                           int          N,
                           int          tileK)
{
  // Each thread.y strides across the tile’s K rows.
  for(int t = threadIdx.y; t < tileK; t += blockDim.y) {
    int r = k0 + t;               // global row
    int c = colBase + threadIdx.x;// global col
    // boundary guard + zero-pad
    float v = 0.0f;
    if(r < K && c < N) {
      v = B[r * N + c];
    }
    // store into shared-mem: col-major within the tile
    tileB[ t * blockDim.x + threadIdx.x ] = v;
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
  load_BTile(buf0+tileM*tileK, B, colBase, 0, K, N, tileK);
  __syncthreads();
  bool useBuf0 = true;  // indicates which buffer holds the “current” data

  // processes A and B in tiles
  for(int k0 = tileK; k0 < K; k0 += tileK) {

    // decide which buffer is computing and which buffer is fetching data
    float* currBuf = useBuf0 ? buf0 : buf1;
    float* nextBuf = useBuf0 ? buf1 : buf0;

    // 1) Prefetch the next tile into nextBuf (no sync)
    load_ATile(nextBuf,A, rowBase, k0, M, K, tileK);
    load_BTile(nextBuf+tileM*tileK, B, colBase, k0,K, N, tileK);

    // 2) Compute on the “current” buffer
    compute_tile(currBuf, tileM, tileK, tileN, acc);

    // 3) barrier: now nextBuf is fully loaded
    __syncthreads();

    // 4) flip buffers
    useBuf0 = !useBuf0;
}
__syncthreads();
float* lastBuf = useBuf0 ? buf0 : buf1;
compute_tile(lastBuf, tileM, tileK, tileN, acc);

int r = rowBase + localRow;
int c = colBase + localCol;
if (r < M && c < N) {
    C[r*N + c] = acc;
    }

}

int main(int argc, char** argv) {

  // Defaults
  int tileM = 32, tileN = 32, tileK = 32;
  int M = 1024, N = 1024, K = 512;

  // Parse overrides
  for(int i = 1; i+1 < argc; i += 2) {
    if(!strcmp(argv[i], "--tile-m")) tileM = std::atoi(argv[i+1]);
    else if(!strcmp(argv[i], "--tile-n")) tileN = std::atoi(argv[i+1]);
    else if(!strcmp(argv[i], "--tile-k")) tileK = std::atoi(argv[i+1]);
  }

  // Clamp
  tileM = std::min(tileM, M);
  tileN = std::min(tileN, N);
  tileK = std::min(tileK, K);

  // Check shared memory budget
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  size_t maxSmem = prop.sharedMemPerBlock;
  size_t bufFloats = size_t(tileM)*tileK + size_t(tileK)*tileN;
  size_t shmemTiled = 2 * bufFloats * sizeof(float);
  assert((bufFloats * sizeof(float)) <= maxSmem);
  assert((shmemTiled) <= maxSmem);

  std::cout<<"Problem: M="<<M<<" N="<<N<<" K="<<K<<"\n"
           <<"Tiles: ("<<tileM<<"×"<<tileK<<" and "<<tileK<<"×"<<tileN<<")\n"
           <<"Shared mem needed (single buf): "<<(bufFloats*sizeof(float))/1024<<" KB\n"
           <<"Shared mem needed (double buf): "<<(shmemTiled)/1024<<" KB\n";

  // Allocate host data
  std::vector<float> hA(M*K), hB(K*N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist;
  for(auto &x: hA) x = dist(rng);
  for(auto &x: hB) x = dist(rng);

  // Allocate device data
  float *dA, *dB, *dC_naive, *dC_tiled;
  hipMalloc(&dA, M*K*sizeof(float));
  hipMalloc(&dB, K*N*sizeof(float));
  hipMalloc(&dC_naive, M*N*sizeof(float));
  hipMalloc(&dC_tiled, M*N*sizeof(float));

  // Copy inputs
  hipMemcpy(dA, hA.data(), M*K*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dB, hB.data(), K*N*sizeof(float), hipMemcpyHostToDevice);

  // Launch configs
  dim3 threads_naive(16,16),  // you can tune this
       grid_naive((N+15)/16,(M+15)/16);

  dim3 threads_tiled(tileN, tileM),
       grid_tiled((N+tileN-1)/tileN,(M+tileM-1)/tileM);

  // Create HIP events
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  const int ITER = 50;

  // Warm-up
  hipLaunchKernelGGL(pairwise_naive,
                     grid_naive, threads_naive, 0, 0,
                     dA,dB,dC_naive,M,N,K);
  hipLaunchKernelGGL(pairwise_tiled,
                     grid_tiled, threads_tiled, shmemTiled, 0,
                     dA,dB,dC_tiled,M,N,K,tileK);
  hipDeviceSynchronize();

  // Time naive
  hipEventRecord(start);
  for(int i=0; i<ITER; ++i) {
    hipLaunchKernelGGL(pairwise_naive,
                       grid_naive, threads_naive, 0, 0,
                       dA,dB,dC_naive,M,N,K);
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
                       dA,dB,dC_tiled,M,N,K,tileK);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms_tiled;
  hipEventElapsedTime(&ms_tiled, start, stop);
  ms_tiled /= ITER;

  // Copy back for a quick correctness check
  std::vector<float> hC_naive(M*N), hC_tiled(M*N);
  hipMemcpy(hC_naive.data(), dC_naive, M*N*sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hC_tiled.data(), dC_tiled, M*N*sizeof(float), hipMemcpyDeviceToHost);

  // Validate
  double maxErr = 0;
  for(int i=0;i<M*N;++i){
    double diff = std::fabs(static_cast<double>(hC_naive[i] - hC_tiled[i]));
    maxErr = std::max<double>(maxErr, diff);
  }
  std::cout<<"Max error between naive/tiled: "<<maxErr<<"\n";

  // Compute GFLOPS (2 FLOPs per multiply-add)
  double flops = double(M)*N*2*K;
  double g_naive = flops / (ms_naive*1e-3) / 1e9;
  double g_tiled = flops / (ms_tiled*1e-3) / 1e9;
  double speedup = ms_naive / ms_tiled;

  std::cout<<"Naive: "<<ms_naive<<" ms  ("<<g_naive<<" GFLOPS)\n"
           <<"Tiled: "<<ms_tiled<<" ms  ("<<g_tiled<<" GFLOPS)\n"
           <<"Speedup: "<<speedup<<"×\n";

  // Cleanup
  hipFree(dA); hipFree(dB);
  hipFree(dC_naive); hipFree(dC_tiled);
  hipEventDestroy(start); hipEventDestroy(stop);

  return 0;
}