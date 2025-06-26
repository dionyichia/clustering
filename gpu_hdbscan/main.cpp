#include <hip/hip_runtime.h>
#include <cstring>   // for strcmp
#include <iostream>
#include <thrust/sort.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <functional>

#include "kd_tree/include/util.hpp"
#include "kd_tree/include/distance.hpp"
#include "kd_tree/include/kdtree.hpp"
#include "kd_tree/include/types.hpp"
#include "boruvka/boruvka.hpp" 
#include "single_linkage/single_linkage.hpp"


// Recursively gather all original points under cluster `c`.
void collect_members(int c,
                     int N_pts,
                     const std::vector<int>& left_child,
                     const std::vector<int>& right_child,
                     std::vector<int>& out)
{
    if (c < N_pts) {
        // leaf
        out.push_back(c);
    } else {
        collect_members(left_child[c], N_pts, left_child, right_child, out);
        collect_members(right_child[c],N_pts, left_child, right_child, out);
    }
}

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
  std::vector<int> pointIndexes(points.size());
  for (int i = 0; i < (int)points.size(); ++i){
        pointIndexes.push_back(static_cast<int>(i));
    }
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

  /* 
     all_edges = flattened array of k*N edges
     pointIndexes = array of indices of points
     numEdges = size of edges
  */ 

  int numEdges = all_edges.size();
  ullong n_vertices = static_cast<ullong>(N);
  ullong n_edges = static_cast<ullong>(numEdges);

  std::cout << "\n=== Running Boruvka MST Algorithm ===" << std::endl;
  std::cout << "Vertices: " << n_vertices << ", Edges: " << n_edges << std::endl;
    
    // Convert std::vector<Edge> to Edge* array for Boruvka function
    Edge* edge_array = all_edges.data();
    
    // Call your Boruvka MST function
    MST result = boruvka_mst(n_vertices, n_edges, edge_array);
    
    std::cout << "\n=== Boruvka MST Results ===" << std::endl;
    std::cout << "MST Weight: " << result.weight << std::endl;
    
    // Print MST edges
    int mst_edge_count = 0;
    std::vector<Edge> mst_edges;

    // Print all Edges take too long
    std::cout << "MST Edges:" << std::endl;
    for (ullong i = 0; i < n_edges; ++i) {
        if (result.mst[i] == 1) {
            mst_edges.push_back(edge_array[i]);
            // std::cout << "Edge " << i << ": (" << edge_array[i].u << "," 
            //         << edge_array[i].v << ") weight=" << edge_array[i].weight << std::endl;
            mst_edge_count++;
        }
    }

    std::cout << "Total MST edges: " << mst_edge_count << std::endl;
    std::cout << "Expected MST edges: " << n_vertices - 1 << std::endl;

    // Post process MST into list of edges

    
    // TODO: Optimise
   int N_pts = points.size();

   std::cout << "\n=== Running Single Linkage Clustering ===" << std::endl;

    // Set min_cluster_size if not already set
    if (min_cluster_size == NULL) {
        min_cluster_size = 2;  // or use k as default
    }

    // Call the single linkage clustering function
    std::vector<std::vector<int>> clusters = single_linkage_clustering(
        mst_edges, 
        N_pts, 
        min_cluster_size
    );

    std::cout << "Single linkage clustering completed." << std::endl;

    /* 
        COPY OUTPUT FROM BORUVKA INTO GPU AND RUN THRUST SORT, SHOULD BE FASTER
        BUT WILL INCUR COPY FROM HOST TO DEVICE COST
    */
    // assume `mst_edges` is your input vector<Edge> of size N–1
    // std::sort(mst_edges.begin(), mst_edges.end(),
    //             [] (const Edge &a, const Edge &b) { return a.weight < b.weight; });

    // // After sorting:
    // assert(!mst_edges.empty());
    // for (auto &e : mst_edges) {
    //     assert(e.u >= 0 && e.u < N_pts);
    //     assert(e.v >= 0 && e.v < N_pts);
    //     assert(e.weight > 0);
    // }
    
    // // int N_pts = /* number of points */;
    // int max_clusters = 2*N_pts;         // All p
    // std::vector<int> parent(max_clusters), sz(max_clusters);
    // std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    // std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // // 2. Guard against zero‐length edges (just in case)
    // float smallest_weight = mst_edges.front().weight;
    // if (smallest_weight <= 0.f) {
    //     // handle degenerate case (e.g. set to a tiny epsilon)
    //     smallest_weight = std::numeric_limits<float>::min();
    // }

    // /* Compute lambda_max as the inverse of smallest_mrd. 
    //    Prevents Singleton Clusters from being most stable.
    // */
    // float lambda_max = 1.f / smallest_weight;
    // /* initialise all points as singleton clusters 
    //    singleton clusters have cluster ids within [0,N_pts-1]
    // */
    // for(int i = 0; i < N_pts; ++i){
    //     parent[i]       = i;
    //     sz[i]           = 1;
    //     birth_lambda[i]= lambda_max;
    //     death_lambda[i]= 0;
    //     stability[i]   = 0;
    // }
    // int next_cluster_id = N_pts;

    // // lambda function for finding parent of a cluster
    // // includes two pass path compression 
    // // first pass to find root
    // // second pass to update all nodes along the path to point to root
    // auto find_root = [&](int x){
    //     // 1) Find the root
    //     int root = x;
    //     while(parent[root] != root)
    //         root = parent[root];
    //     // 2) Compress the path
    //     while(parent[x] != root){
    //         int next = parent[x];
    //         parent[x] = root;
    //         x = next;
    //     }
    //     return root;
    // };

    // // lambda function for finding root node of vertex in an edge 
    // for(auto &e : mst_edges){
    //     int c1 = find_root(e.u),
    //         c2 = find_root(e.v);
    //     // if clusters are already connected, continue
    //     if(c1 == c2){
    //         continue;
    //     }

    //     // else make new cluster
    //     float lambda = 1.f / e.weight;
    //     // record death of c1, c2
    //     death_lambda[c1] = lambda;
    //     death_lambda[c2] = lambda;
    //     // update their stability contributions
    //     stability[c1] += (birth_lambda[c1] - death_lambda[c1]) * sz[c1];
    //     stability[c2] += (birth_lambda[c2] - death_lambda[c2]) * sz[c2];

    //     // make new cluster
    //     int c_new = next_cluster_id++;
    //     parent[c1] = parent[c2] = c_new;
    //     parent[c_new] = c_new;
    //     sz[c_new]         = sz[c1] + sz[c2];
    //     birth_lambda[c_new] = lambda;
    //     stability[c_new]   = 0;
    //     death_lambda[c_new] = 0;
    //     left_child[c_new]  = c1;
    //     right_child[c_new] = c2;
    // }

    // // initialise all remaining singleton clusters to die at lambda = 0
    // for(int c = 0; c < next_cluster_id; ++c){
    //     if(parent[c] == c){
    //         death_lambda[c] = 0;
    //         stability[c]   += (birth_lambda[c] - death_lambda[c]) * sz[c];
    //     }
    // }

    // // 1) Build a list of candidate cluster IDs:
    // std::vector<int> candidates;
    // for(int c = 0; c < next_cluster_id; ++c) {
    //     if (sz[c] >= min_cluster_size && death_lambda[c] > 0)  // <-- drop the root
    //         candidates.push_back(c);
    // }


    // // 2) Sort them by descending stability:
    // std::sort(candidates.begin(), candidates.end(),
    //       [&](int a, int b){
    //           return stability[a] > stability[b];
    //       });

    
    // // 3a) Pick the “most stable” clusters, skipping any whose direct children
    // //     are already selected
    // //     final clusters are built in descending stability as well
    // std::vector<bool> is_selected(max_clusters, false);
    // std::vector<int> final_clusters;
    // for(int c : candidates) {
    //     int L = left_child[c], R = right_child[c];
    //     // if either child is itself a selected cluster, skip this one
    //     if ((L >= N_pts && is_selected[L]) ||
    //         (R >= N_pts && is_selected[R])) {
    //     continue;
    //     }
    //     // otherwise select it
    //     is_selected[c] = true;
    //     final_clusters.push_back(c);
    // }
    // // 3b) Now assign each point to the first (most-stable) selected cluster it appears in
    // std::vector<int> assignment(N_pts, -1);
    // std::vector<std::vector<int>> clusters;
    // for(int c : final_clusters){
    //     // gather all member points under cluster c
    //     std::vector<int> mem;
    //     collect_members(c, N_pts, left_child, right_child, mem);

    //     // build this cluster’s final list of *newly* assigned points
    //     std::vector<int> this_cluster;
    //     for(int p : mem){
    //         if(assignment[p] == -1){
    //             assignment[p] = clusters.size();
    //             this_cluster.push_back(p);
    //         }
    //     }
    //     if(!this_cluster.empty())
    //         clusters.push_back(std::move(this_cluster));
    // }

    // 4) Output
    std::cout << "Found " << clusters.size()
            << " clusters (min size = " << min_cluster_size << ")\n";
    for(size_t i = 0; i < clusters.size(); ++i){
        std::cout << "Cluster " << i << " (" 
                << clusters[i].size() << " points): ";
        for(int p : clusters[i])
            std::cout << p << " ";
        std::cout << "\n";
    }

    // Clean up
    if (result.mst) {
        free(result.mst);
    }
}
