#include "mrd.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

__global__ void computeMRD(
    const float* __restrict__ coreDist,
    const float* __restrict__ dist,
    float* mrd,
    int   M)
{
    // global thread coords
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // only process the strict upper triangle
    if (i < M && j < M && i < j) {
        float cd_i = coreDist[i];
        float cd_j = coreDist[j];
        float d    = dist[i * M + j];
        float mr   = fmaxf( fmaxf(cd_i, cd_j), d );

        // write both (i,j) and (j,i)
        mrd[i * M + j] = mr;
        mrd[j * M + i] = mr;
    }
}

std::vector<float> compute_mrd(
    const std::vector<float>& coreDist,
    const std::vector<float>& distance,
    int M)
{
  float *d_coreDist, *d_dist, *d_mrd;
  hipMalloc(&d_coreDist, (size_t)M * sizeof(float));
  hipMalloc(&d_dist,     (size_t)M * M * sizeof(float));
  hipMalloc(&d_mrd,      (size_t)M * M * sizeof(float));

  hipMemcpy(d_coreDist,
            coreDist.data(),
            (size_t)M * sizeof(float),
            hipMemcpyHostToDevice);

  hipMemcpy(d_dist,
            distance.data(),
            (size_t)M * M * sizeof(float),
            hipMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid( (M + block.x - 1) / block.x,
            (M + block.y - 1) / block.y );

  hipLaunchKernelGGL(
      computeMRD,
      grid, block,
      0,    // no shared-mem
      0,    // default stream
      d_coreDist,
      d_dist,
      d_mrd,
      M
  );
  hipDeviceSynchronize();

  // 4) Copy the MRD matrix back to host
  std::vector<float> h_mrd((size_t)M * M);
  hipMemcpy(h_mrd.data(),
            d_mrd,
            (size_t)M * M * sizeof(float),
            hipMemcpyDeviceToHost);
  for (int i = 0; i < M; ++i) {
        h_mrd[i*M + i] = coreDist[i];
  }
//   // 5) (Optional) inspect or print, e.g. the first row
//   for(int j = 0; j < std::min(M, 8); ++j) {
//       std::cout << "mrd[0,"<< j <<"] = "
//                 << h_mrd[0 * M + j] << "\n";
//   }

  // 6) Cleanup
  hipFree(d_coreDist);
  hipFree(d_dist);
  hipFree(d_mrd);
  return h_mrd;
}
