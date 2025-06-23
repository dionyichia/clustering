// core_distance.hpp
#pragma once
#include <vector>

// Compute the “core distance” for each of M points in an M×K dataset.
// - `data`: length-M*K row-major array
// - returns: length-M vector of the k-th smallest neighbor distance.
std::vector<float> compute_distances_gpu(
    const std::vector<float>& hA,
    int M,
    int K,
    int tileM,
    int tileK);

std::vector<float> compute_core_distances(
    const std::vector<float>& distMatrix,
    int M,
    int minPts); 