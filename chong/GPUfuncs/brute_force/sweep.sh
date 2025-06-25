#!/usr/bin/env bash
# sweep.sh

# define the set of Ms and Ks you want to test
Ms=(512 1024 2048 4096,8192,16384)
Ks=(2)

# optional: choose fixed tile sizes and minPts
TILE_M=32
TILE_K=32
MINPTS=5

mkdir -p results

for M in "${Ms[@]}"; do
  for K in "${Ks[@]}"; do
    echo "Running M=$M, K=$K …"
    ./pairwise \
      --M "$M" --K "$K" \
      --tile-m "$TILE_M" --tile-k "$TILE_K" \
      --minpts "$MINPTS" \
      > results/M${M}_K${K}.log 2>&1
  done
done