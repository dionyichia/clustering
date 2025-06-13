#include "core_dist.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <assert.h>
#include <cassert>
#include <random>

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
  auto cd = compute_core_distances_gpu(data, M, K, tileM, tileK, minPts);
  // … do something with cd …
  return 0;

}