// mrd.hpp
#include <vector>

// Compute the mutual reachability distance for each of M points in an M×K dataset.
// - coreDist: M*M matrix of core distances for each point
// - dist: M*M matrix of distances between each point
// - M: number of data points (rows)
// - returns: length-M vector, with each points MRD.
std::vector<float> compute_mrd(
    const std::vector<float>& coreDist,
    const std::vector<float>& distance,
    int M);