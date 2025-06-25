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
  float p = 2.0f
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
