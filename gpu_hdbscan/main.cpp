#include <hip/hip_runtime.h>
#include <cstring>   // for strcmp
#include <iostream>
#include <thrust/sort.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "kd_tree/include/util.hpp"
#include "kd_tree/include/distance.hpp"
#include "kd_tree/include/kdtree.hpp"
#include "kd_tree/include/types.hpp"
#include "boruvka/boruvka.hpp" 
#include "single_linkage/single_linkage.hpp"

// QUIET MODE to silence debug statements
bool quiet_mode = false;

// Replace all std::cout statements with conditional output:
#define DEBUG_PRINT(x) if (!quiet_mode) { std::cout << x; }

void outputClusterLabels(const std::vector<std::vector<int>>& clusters, int total_points) {
    // Create label array initialized to -1 (noise)
    std::vector<int> labels(total_points, -1);
    
    // Assign cluster labels
    for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (int point_id : clusters[cluster_id]) {
            if (point_id >= 0 && point_id < total_points) {
                labels[point_id] = cluster_id;
            }
        }
    }
    
    // ALWAYS output cluster labels (needed for Python parsing)
    std::cout << "CLUSTER_LABELS:";
    for (int i = 0; i < labels.size(); ++i) {
        std::cout << " " << labels[i];
    }
    std::cout << std::endl;
    
    // Output cluster statistics (conditional)
    DEBUG_PRINT("CLUSTER_STATS:" << std::endl);
    DEBUG_PRINT("  Total points: " << total_points << std::endl);
    DEBUG_PRINT("  Number of clusters: " << clusters.size() << std::endl);
    int noise_count = std::count(labels.begin(), labels.end(), -1);
    DEBUG_PRINT("  Noise points: " << noise_count << std::endl);
    DEBUG_PRINT("  Clustered points: " << (total_points - noise_count) << std::endl);
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
              DEBUG_PRINT( "Number Of Dimensions " << dimensions << "\n");
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--minpts")) {
          try{
              k = std::stoi(argv[i+1]);
              i += 2;
              DEBUG_PRINT("Minimum Points " << k << "\n"); 
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
              DEBUG_PRINT( "Read " << points.size() << " points." << "\n");
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
              DEBUG_PRINT( "Distance Metric Selected: " << metricName(metric) << "\n");
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
              DEBUG_PRINT( "Minkowski P Value: " << minkowskiP << "\n");
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
              DEBUG_PRINT( "Min Cluster Size: " << min_cluster_size << "\n");
              i += 2;
          } catch(const std::exception& e) {
              std::cerr << e.what() << "\n";
              printUsage(argv[0]);
              return 1;
          }
      }
      else if (!strcmp(argv[i], "--quiet") || !strcmp(argv[i], "-q")) {
            quiet_mode = true;
            DEBUG_PRINT("Quiet mode enabled" << std::endl);  // This won't print since quiet_mode is now true
            i += 1;  // Only increment by 1 since there's no argument after --quiet
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

  if (!quiet_mode) {
        printAndVerifyCoreDists(points, core_dist, k, metric, minkowskiP);
    }


  // Calculate Mutual Reachability Distance
  convertToMutualReachability(knn_graph, core_dist);
  if (!quiet_mode) {
        printAndVerifyMutualReachability(points, core_dist, knn_graph, metric, minkowskiP);
    }
  // After you've built your knn_graph and converted to mutual reachability
  std::vector<Edge> all_edges = flatten(knn_graph);
  // Now you can sort by weight if needed
  constexpr size_t GPU_SORT_THRESHOLD = 1'000'000;
  std::cout << "[DEBUG] Before sort, first few weights:";
  for (size_t i = 0; i < std::min<size_t>(5, all_edges.size()); ++i)
        std::cout << " " << all_edges[i].weight;
    std::cout << "\n";

    if (all_edges.size() >= GPU_SORT_THRESHOLD) {
        // copy up to GPU
        thrust::device_vector<Edge> d_edges(all_edges.begin(), all_edges.end());
        // GPU parallel sort using Edge::operator<
        thrust::sort(d_edges.begin(), d_edges.end());
        // copy back
        thrust::copy(d_edges.begin(), d_edges.end(), all_edges.begin());
        std::cout << "[DEBUG] Used thrust::sort on GPU\n";
    } else {
        std::sort(all_edges.begin(), all_edges.end());
        std::cout << "[DEBUG] Used std::sort on CPU\n";
    }
    std::cout << "[DEBUG] After sort, first few weights:";
    for (size_t i = 0; i < std::min<size_t>(5, all_edges.size()); ++i)
        std::cout << " " << all_edges[i].weight;
    std::cout << "\n";
  DEBUG_PRINT("Total edges: " << all_edges.size() << std::endl);
  if (!quiet_mode) {
        printFirstNEdges(all_edges);
    }

  /* 
     all_edges = flattened array of k*N edges
     pointIndexes = array of indices of points
     numEdges = size of edges
  */ 

  int numEdges = all_edges.size();
  ullong n_vertices = static_cast<ullong>(N);
  ullong n_edges = static_cast<ullong>(numEdges);

  DEBUG_PRINT( "\n=== Running Boruvka MST Algorithm ===" << "\n");
  DEBUG_PRINT( "Vertices: " << n_vertices << ", Edges: " << n_edges << "\n");
    
    // Convert std::vector<Edge> to Edge* array for Boruvka function
    Edge* edge_array = all_edges.data();
    
    // Call your Boruvka MST function
    MST result = boruvka_mst(n_vertices, n_edges, edge_array);
    
    DEBUG_PRINT( "\n=== Boruvka MST Results ===" << "\n");
    DEBUG_PRINT( "MST Weight: " << result.weight << "\n");
    
    // Print MST edges
    int mst_edge_count = 0;
    std::vector<Edge> mst_edges;

    // Print all Edges take too long
    DEBUG_PRINT( "MST Edges:" << "\n");
    for (ullong i = 0; i < n_edges; ++i) {
        if (result.mst[i] == 1) {
            mst_edges.push_back(edge_array[i]);
            // DEBUG_PRINT( "Edge " << i << ": (" << edge_array[i].u << "," 
            //         << edge_array[i].v << ") weight=" << edge_array[i].weight << "\n");
            mst_edge_count++;
        }
    }

    DEBUG_PRINT( "Total MST edges: " << mst_edge_count << "\n");
    DEBUG_PRINT( "Expected MST edges: " << n_vertices - 1 << "\n");

    // Post process MST into list of edges

    
    // TODO: Optimise
    int N_pts = points.size();
    DEBUG_PRINT( "[DEBUG] Number of points (N_pts): " << N_pts << "\n");


   DEBUG_PRINT( "\n=== Running Single Linkage Clustering ===" << "\n");

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

    DEBUG_PRINT( "Single linkage clustering completed." << "\n");


    // Output cluster labels for Python parsing
    outputClusterLabels(clusters, N_pts);

    // Clean up
    if (result.mst) {
        free(result.mst);
        DEBUG_PRINT( "[DEBUG] Freed MST memory \n");
    }

    return 0;  // Add return statement
}

    // // Set min_cluster_size if not already set
    // if (min_cluster_size == NULL) {
    //     min_cluster_size = 2;  // or use k as default
    // }

    // // Call the single linkage clustering function
    // std::vector<std::vector<int>> clusters = single_linkage_clustering(
    //     mst_edges, 
    //     N_pts, 
    //     min_cluster_size
    // );

    // DEBUG_PRINT( "Single linkage clustering completed." << "\n");

    // /* 
    //     COPY OUTPUT FROM BORUVKA INTO GPU AND RUN THRUST SORT, SHOULD BE FASTER
    //     BUT WILL INCUR COPY FROM HOST TO DEVICE COST
    // */
    // // assume `mst_edges` is your input vector<Edge> of size N–1
    // DEBUG_PRINT( "[DEBUG] Before sort, first few weights:";
    // for (int i = 0; i < std::min<size_t>(5, mst_edges.size()); ++i)
    //     DEBUG_PRINT( " " << mst_edges[i].weight;
    // DEBUG_PRINT( "\n";

    // std::sort(mst_edges.begin(), mst_edges.end(),
    //             [] (const Edge &a, const Edge &b) { return a.weight < b.weight; });

    // DEBUG_PRINT( "[DEBUG] After sort, smallest 5 weights:";
    // for (int i = 0; i < std::min<size_t>(5, mst_edges.size()); ++i)
    //     DEBUG_PRINT( " " << mst_edges[i].weight;
    // DEBUG_PRINT( "\n";

    // // After sorting:
    // assert(!mst_edges.empty());
    // for (auto &e : mst_edges) {
    //     assert(e.u >= 0 && e.u < N_pts);
    //     assert(e.v >= 0 && e.v < N_pts);
    //     assert(e.weight > 0);
    // }
    // DEBUG_PRINT( "[DEBUG] Edge assertions passed.\n";

    // int max_clusters = 2 * N_pts;
    // std::vector<int> parent(max_clusters), sz(max_clusters);
    // std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    // std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);

    // float smallest_weight = mst_edges.front().weight;
    // DEBUG_PRINT( "[DEBUG] Raw smallest weight: " << smallest_weight << "\n";
    // if (smallest_weight <= 0.f) {
    //     smallest_weight = std::numeric_limits<float>::min();
    //     DEBUG_PRINT( "[DEBUG] Adjusted smallest weight to epsilon: " << smallest_weight << "\n";
    // }

    // float lambda_max = 1.f / smallest_weight;
    // DEBUG_PRINT( "[DEBUG] lambda_max: " << lambda_max << "\n";

    // /* initialise all points as singleton clusters */
    // for(int i = 0; i < N_pts; ++i){
    //     parent[i]        = i;
    //     sz[i]            = 1;
    //     birth_lambda[i]  = lambda_max;
    //     death_lambda[i]  = 0;
    //     stability[i]     = 0;
    // }
    // int next_cluster_id = N_pts;
    // DEBUG_PRINT( "[DEBUG] Initialized " << next_cluster_id << " singleton clusters\n";

    // // lambda to find root (path-compressed):
    // auto find_root = [&](int x){
    //     int root = x;
    //     while(parent[root] != root) root = parent[root];
    //     while(parent[x] != root){
    //         int next = parent[x];
    //         parent[x] = root;
    //         x = next;
    //     }
    //     return root;
    // };

    // // Build hierarchy
    // for(auto &e : mst_edges){
    //     int c1 = find_root(e.u), c2 = find_root(e.v);
    //     if(c1 == c2) continue;

    //     float lambda = 1.f / e.weight;
    //     death_lambda[c1] = death_lambda[c2] = lambda;
    //     stability[c1] += (birth_lambda[c1] - lambda) * sz[c1];
    //     stability[c2] += (birth_lambda[c2] - lambda) * sz[c2];

    //     int c_new = next_cluster_id++;
    //     parent[c1] = parent[c2] = c_new;
    //     parent[c_new] = c_new;
    //     sz[c_new]           = sz[c1] + sz[c2];
    //     birth_lambda[c_new] = lambda;
    //     stability[c_new]    = 0;
    //     death_lambda[c_new] = 0;
    //     left_child[c_new]   = c1;
    //     right_child[c_new]  = c2;

    //     DEBUG_PRINT( "[DEBUG] Merged clusters " << c1 << " and " << c2
    //             << " into " << c_new << " at lambda=" << lambda << "\n";
    // }

    // DEBUG_PRINT( "[DEBUG] Total clusters created: " << next_cluster_id << "\n";

    // // Finalize singleton deaths
    // for(int c = 0; c < next_cluster_id; ++c){
    //     if(parent[c] == c){
    //         death_lambda[c] = 0;
    //         stability[c]   += (birth_lambda[c] - 0) * sz[c];
    //     }
    // }

    // // Collect candidates
    // std::vector<int> candidates;
    // for(int c = 0; c < next_cluster_id; ++c){
    //     if(sz[c] >= min_cluster_size && death_lambda[c] > 0)
    //         candidates.push_back(c);
    // }
    // DEBUG_PRINT( "[DEBUG] Number of candidate clusters (size>=" << min_cluster_size
    //         << "): " << candidates.size() << "\n";

    // // Sort candidates by stability
    // std::sort(candidates.begin(), candidates.end(),
    //     [&](int a, int b){ return stability[a] > stability[b]; });
    // DEBUG_PRINT( "[DEBUG] Top 5 candidate stabilities:";
    // for (int i = 0; i < std::min<int>(5, candidates.size()); ++i)
    //     DEBUG_PRINT( " (" << candidates[i] << ":" << stability[candidates[i]] << ")";
    // DEBUG_PRINT( "\n";

    // // Select final clusters
    // std::vector<bool> is_selected(max_clusters, false);
    // std::vector<int> final_clusters;
    // for(int c : candidates) {
    //     int L = left_child[c], R = right_child[c];
    //     if ((L >= N_pts && is_selected[L]) || (R >= N_pts && is_selected[R])) {
    //         DEBUG_PRINT( "[DEBUG] Skipping cluster " << c << " because child already selected\n";
    //         continue;
    //     }
    //     is_selected[c] = true;
    //     final_clusters.push_back(c);
    //     DEBUG_PRINT( "[DEBUG] Selected cluster " << c << "\n";
    // }
    // DEBUG_PRINT( "[DEBUG] Total final clusters: " << final_clusters.size() << "\n";

    // // Assign points
    // std::vector<int> assignment(N_pts, -1);
    // std::vector<std::vector<int>> clusters;
    // for(int c : final_clusters){
    //     std::vector<int> mem;
    //     collect_members(c, N_pts, left_child, right_child, mem);
    //     std::vector<int> this_cluster;
    //     for(int p : mem){
    //         if(assignment[p] == -1){
    //             assignment[p] = clusters.size();
    //             this_cluster.push_back(p);
    //         }
    //     }
    //     if(!this_cluster.empty()){
    //         clusters.push_back(std::move(this_cluster));
    //         DEBUG_PRINT( "[DEBUG] Cluster " << (clusters.size()-1)
    //                 << " got " << clusters.back().size() << " points\n";
    //     }
    // }

    // // 4) Output
    // DEBUG_PRINT( "Found " << clusters.size()
    //         << " clusters (min size = " << min_cluster_size << ")\n";
    // for(size_t i = 0; i < clusters.size(); ++i){
    //     DEBUG_PRINT( "Cluster " << i << " (" 
    //             << clusters[i].size() << " points): ";
    //     for(int p : clusters[i])
    //         DEBUG_PRINT( p << " ";
    //     DEBUG_PRINT( "\n";
    // }