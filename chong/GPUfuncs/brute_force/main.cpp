#include "core_dist.hpp"
#include "mrd.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>
#include <random>
#include <chrono>
// CPU MRD
std::vector<float> compute_mrd_cpu(
    const std::vector<float>& coreDist,
    const std::vector<float>& dist,
    int M)
{
    std::vector<float> mrd(M * M);
    for (int i = 0; i < M; ++i) {
        float cd_i = coreDist[i];
        for (int j = 0; j < M; ++j) {
            float cd_j    = coreDist[j];
            float d_ij    = dist[i * M + j];
            mrd[i * M + j] = std::max(std::max(cd_i, cd_j), d_ij);
        }
    }
    return mrd;
}

int main(int argc, char** argv){
  // dimensions of tiles used in tiling Blocks
  int tileM,tileK;
  // minPts input to HDBSCAN 
  int minPts;
  // M == rows of input A, number of datapoints
  // K == columns of input A, number of features in one datapoint
  int M,K;

  // Parse param overrides (if any)
  int i = 1;
  while (i + 1 < argc) {
      if (!strcmp(argv[i], "--tile-k")) {
        if(std::atoi(argv[i+1]) <= 128){
          tileK  = std::atoi(argv[i+1]);
          i += 2;               // consumed flag + value
        }
        else{
          std::cout<<"Max Tile Dimension is 128"<<"\n";
          return 1;
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
  tileM=32;
  tileK = std::min(tileK, K);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  size_t maxSmem = prop.sharedMemPerBlock;
  size_t bufFloats = size_t(tileM)*tileK + size_t(tileM)*tileK;
  size_t shmemTiled = 2 * bufFloats * sizeof(float);
  assert((bufFloats * sizeof(float)) <= maxSmem);
  assert((shmemTiled) <= maxSmem);
    // Allocate host data
  std::vector<float> data(M*K);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist;
  for(auto &x: data) x = dist(rng);
  // parse M,K,tileM,tileK,minPts, fill a vector `data`
  std::cout<< "Calculating Core Distances..."<<"\n";
  auto distance = compute_distances_gpu(data, M, K, tileM, tileK);
  auto coreDist = compute_core_distances(distance,M,minPts);
  std::cout<< "Finished Calculating Core Distances!"<<"\n";
  // for(int i =0; i < M; i++){
  //   std::cout<<"Point: "<<i<<"Core Distance"<<coreDist[i]<<"\n";
  // }
  // --- GPU MRD + timing ---
  std::cout<< "Calculating MRD using GPU..."<<"\n";
  auto t0 = std::chrono::high_resolution_clock::now();
  auto mrd_gpu = compute_mrd(coreDist, distance, M);
  auto t1 = std::chrono::high_resolution_clock::now();
  double gpu_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
  std::cout<< "Finished Calculating MRD using GPU!"<<"\n";

  // --- CPU MRD + timing ---
  std::cout<< "Calculating MRD using CPU..."<<"\n";
  auto t2 = std::chrono::high_resolution_clock::now();
  auto mrd_cpu = compute_mrd_cpu(coreDist, distance, M);
  auto t3 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double,std::milli>(t3 - t2).count();
  std::cout<< "Finished Calculating MRD using CPU!"<<"\n";
  
  // // --- verify correctness ---
  // const float eps = 1e-5f;
  // bool ok = true;
  // for (size_t idx = 0; idx < mrd_gpu.size(); ++idx) {
  //     if (std::fabs(mrd_gpu[idx] - mrd_cpu[idx]) > eps) {
  //         std::cerr << "Mismatch at idx " << idx
  //                   << ": gpu=" << mrd_gpu[idx]
  //                   << " cpu=" << mrd_cpu[idx] << "\n";
  //         ok = false;
  //         break;
  //     }
  // }
  // if (ok) std::cout << "✔ GPU and CPU MRD match within " << eps << "\n";
  // else    std::cout << "✖ MRD mismatch!\n";

  // --- print timings ---
  std::cout << "GPU MRD time: " << gpu_ms << " ms\n";
  std::cout << "CPU MRD time: " << cpu_ms << " ms\n";

  return 0;

}