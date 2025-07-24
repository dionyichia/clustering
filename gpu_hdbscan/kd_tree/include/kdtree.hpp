#pragma once
#include "types.hpp"
#include "distance.hpp"
#include <queue>

struct KDNode {
  Point             point;
  int               axis;
  int               index;
  std::unique_ptr<KDNode> left, right;
  KDNode(Point pt, int ax, int idx);
};

// recursive helper on a vector<PI>
std::unique_ptr<KDNode> buildKDTreeRecursive(
    std::vector<PI>&   arr, // array of vectors stored as ((vector), index)
    int                l, 
    int                r,
    int                depth
);

// Facade Interface for easier use
std::unique_ptr<KDNode> buildKDTree(std::vector<Point> pts);

/// Query k-nearest neighbors into a max-heap
void queryKNN(
  const KDNode* node,
  const Point&  query,
  int           query_idx,
  int           k,
  std::priority_queue<std::pair<double,int>>& heap,
  const std::vector<Point>& points,
  DistanceMetric metric,
  float p,
  const std::vector<double>*                     weights = nullptr
);

// Replace tuple of each k-nn to every point with MRD instead of distance to point
void convertToMutualReachability(
  std::vector<std::vector<std::pair<int,double>>>& knn_graph,
  const std::vector<double>& core_dist
);

// Sanity Checks
void printAndVerifyCoreDists(
  const std::vector<Point>& points,
  const std::vector<double>& core_dist,
  int k,
  DistanceMetric metric,
  float p = 2.0f
);
void printAndVerifyMutualReachability(
  const std::vector<Point>& points,
  const std::vector<double>& core_dist,
  const std::vector<std::vector<std::pair<int,double>>>& knn_graph,
  DistanceMetric metric,
  float p = 2.0f
);

/**
 * Flatten the k-NN graph into a vector of Edge structs.
 * Each point i has k neighbors, so we get k*N total edges.
 * 
 * @param knn_graph: Vector where knn_graph[i] contains k nearest neighbors of point i
 *                   as pairs of (neighbor_index, mutual_reachability_distance)
 * @return: Vector of Edge structs, each containing (u, v, weight) where:
 *          - u is the source point index
 *          - v is the neighbor point index  
 *          - weight is the mutual reachability distance
 */
std::vector<Edge> flatten(const std::vector<std::vector<std::pair<int,double>>>& knn_graph);

/**
 * Print only the first N edges for quick inspection
 */
void printFirstNEdges(const std::vector<Edge>& edges, int n = 10) ;