#include "util.hpp"
#include "distance.hpp"
#include "kdtree.hpp"
#include <hip/hip_runtime.h>
#include <cstring>   // for strcmp
#include <iostream>

int main(int argc, char** argv) {
  std::vector<Point> points;
  int dimensions = NULL;
  int k = NULL;
  int min_cluster_size = NULL;
  int metricChoice = NULL;
  float minkowskiP = NULL;
  DistanceMetric metric;
  /* Param Overrides 
      --dimensions : number of dimensions (features) of each data point
      --minpts (int): used in calculating core-distance and MRD. Doubles as --minclustersize for cluster extraction
      --input (string): name of file storing input data
      --distanceMetric (int): Choose Distance Metric [1: Manhattan, 2: Euclidean, 3: Chebyshev, 4:Minkowski]
      --minkowskiP (float): P-value for minkowski
      --minclustersize (int): used in cluster extraction
  */
  int i = 1;
  while (i + 1 < argc) {
      if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
          printUsage(argv[0]);
          return 1;
      }
      else if(!strcmp(argv[i], "--dimensions")){
          try{
              dimensions = std::stoi(argv[i+1]);
              i += 2;
              std::cout << "Number Of Dimensions " << dimensions << "\n";
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--minpts")) {
          try{
              k = std::stoi(argv[i+1]);
              i += 2;
              std::cout << "Minimum Points " << k << "\n";
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--input")) {
          try{
              points = readPointsFromFile(argv[i+1],dimensions);
              normalizePoints(points);
              i += 2;
              std::cout << "Read " << points.size() << " points.\n";
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--distMetric")){
          try{
              metricChoice = std::stoi(argv[i+1]);
              switch (metricChoice){
                  case(1):
                      metric = DistanceMetric::Manhattan;
                      break;
                  case(2):
                      metric = DistanceMetric::EuclideanSquared;
                      break;
                  case(3):
                      metric = DistanceMetric::Chebyshev;
                      break;
                  case(4):
                      metric = DistanceMetric::Minkowski;
                      break;
              }
              std::cout << "Distance Metric Selected: " << metricName(metric) << "\n";
              i += 2; 
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--minkowskiP")){
          try{
              minkowskiP = std::stof(argv[i+1]);
              std::cout << "Minkowski P Value: " << minkowskiP << "\n";
              i += 2;
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--minclustersize")){
          try{
              min_cluster_size = std::stoi(argv[i+1]);
              std::cout << "Min Cluster Size: " << min_cluster_size << "\n";
              i += 2;
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else {
          // unrecognized flag: skip just the flag
          std::cerr << "Warning: unknown option '" << argv[i] << "\n";
          printUsage(argv[0]);
          i += 1;
      }
  }
  if (k == NULL){
      k = 2;
  }
  if (dimensions == NULL){
      std::cerr << "Dimensions Of Data Not Provided" << "\n";
      printUsage(argv[0]);
      return 1;
  }
  if (metric == DistanceMetric::Minkowski && minkowskiP == NULL){
      std::cerr << "P-Value not provided" << "\n";
      printUsage(argv[0]);
      return 1;
  }


  int N = points.size();
  std::vector<std::vector<std::pair<int,double>>> knn_graph(points.size());
  std::vector<double> core_dist(points.size());

  auto root = buildKDTree(points);

  for (int i = 0; i < N; ++i) {
      // 1) Prepare an empty max-heap
      std::priority_queue<std::pair<double,int>> heap;

      // 2) Query the tree for point i
      queryKNN(root.get(), points[i], i, k, heap, points,metric,minkowskiP);
      // 3) Extract neighbors (and record the core distance)
      //    Since heap is max‐heap, after you pop k elements,
      //    the last popped distance = core distance
      double d_k = 0;
      // 1) Record core‐distance before you empty the heap
      double coreDist = heap.top().first;
      if(metric == DistanceMetric::EuclideanSquared){
          core_dist[i] = std::sqrt(coreDist);
      }
      else{
          core_dist[i] = coreDist;
      }
      std::vector<std::pair<int,double>> nbrs;
      nbrs.reserve(k);
      while (!heap.empty()) {
      auto [d_sq, idx] = heap.top(); heap.pop();
      if(metric == DistanceMetric::EuclideanSquared){
          nbrs.emplace_back(idx, std::sqrt(d_sq));
      }
      else{
          nbrs.emplace_back(idx, d_sq);
      }
      }
      // reverse to have them in ascending order if you like
      std::reverse(nbrs.begin(), nbrs.end());
      knn_graph[i] = std::move(nbrs);
  }
  printAndVerifyCoreDists(points, core_dist, k,metric,minkowskiP);

  // Calculate Mutual Reachability Distance
  convertToMutualReachability(knn_graph, core_dist);
  printAndVerifyMutualReachability(points, core_dist, knn_graph,metric,minkowskiP);
  // After you've built your knn_graph and converted to mutual reachability
  std::vector<Edge> all_edges = flatten(knn_graph);
  // Now you can sort by weight if needed
  std::sort(all_edges.begin(), all_edges.end()); // Uses your Edge::operator
  std::cout << "Total edges: " << all_edges.size() << std::endl;
  printFirstNEdges(all_edges);
}

