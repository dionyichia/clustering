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
#include <iomanip>

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
    // std::cout << "CLUSTER_LABELS:";
    // for (int i = 0; i < labels.size(); ++i) {
    //     std::cout << " " << labels[i];
    // }
    // std::cout << std::endl;
    
    // Output cluster statistics (conditional)
    DEBUG_PRINT("CLUSTER_STATS:" << std::endl);
    DEBUG_PRINT("  Total points: " << total_points << std::endl);
    DEBUG_PRINT("  Number of clusters: " << clusters.size() << std::endl);
    int noise_count = std::count(labels.begin(), labels.end(), -1);
    DEBUG_PRINT("  Noise points: " << noise_count << std::endl);
    DEBUG_PRINT("  Clustered points: " << (total_points - noise_count) << std::endl);
}

int main(int argc, char** argv) {
  std::vector<int> ground_truth_labels;
  std::vector<Point> points;
  std::set<int> skip_columns;
  std::string input_filename;
  int noBenchMark = 0;
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
       else if (!strcmp(argv[i], "--skip-toa")) {
        skip_columns.insert(0);  // TOA is typically column 0
        i += 1;
        DEBUG_PRINT("Will skip TOA column (index 0)\n");
      }
      else if (!strcmp(argv[i], "--skip-amp")) {
        skip_columns.insert(3);  // Amp_S0 is typically column 3 based on your data format
        i += 1;
        DEBUG_PRINT("Will skip Amp column (index 3)\n");
      }
      else if (!strcmp(argv[i], "--skip-columns")) {
        // Parse comma-separated list of column indices to skip
        try {
            std::string cols_str = argv[i+1];
            std::stringstream ss(cols_str);
            std::string col;
            while (std::getline(ss, col, ',')) {
                int col_idx = std::stoi(col);
                skip_columns.insert(col_idx);
                DEBUG_PRINT("Will skip column index " << col_idx << "\n");
            }
            i += 2;
        } catch(const std::exception& e) {
            std::cerr << "Error parsing skip columns: " << e.what() << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
      else if (!strcmp(argv[i], "--input")) {
        try{
            input_filename = argv[i+1];  // Store filename for later use
            i += 2;
            DEBUG_PRINT("Input file: " << input_filename << "\n");
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
      else if (!strcmp(argv[i], "--noBenchMark")){
        noBenchMark = 1;
        i += 1;
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
    if (input_filename.empty()) {
        std::cerr << "Input file not specified\n";
        printUsage(argv[0]);
        return 1;
    }
    try {
        if(noBenchMark){
            points = readPointsFromFile(input_filename, dimensions, ground_truth_labels, skip_columns);
        }
        else{
            points = readPointsFromFile(input_filename, dimensions, skip_columns);
        }
        normalizePoints(points);
        DEBUG_PRINT("Read " << points.size() << " points with " << dimensions 
                    << " dimensions (skipped " << skip_columns.size() << " columns).\n");
        
        // Print which columns were skipped
        if (!skip_columns.empty() && !quiet_mode) {
            DEBUG_PRINT("Skipped columns: ");
            for (int col : skip_columns) {
                DEBUG_PRINT(col << " ");
            }
            DEBUG_PRINT("\n");
        }
    } catch(const std::exception& e) {
        std::cerr << e.what() << "\n";
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


    if (!ground_truth_labels.empty() && ground_truth_labels.size() == N_pts) {
        ClusterMetrics metrics = evaluateClustering(ground_truth_labels, clusters, N_pts);
        printClusteringEvaluation(metrics, quiet_mode);
    } else {
        DEBUG_PRINT("Warning: Ground truth labels not available or size mismatch. Skipping evaluation." << "\n");
        DEBUG_PRINT("Ground truth size: " << ground_truth_labels.size() << ", Points size: " << N_pts << "\n");
    }
    // Output cluster labels for Python parsing
    outputClusterLabels(clusters, N_pts);

    // Clean up
    if (result.mst) {
        free(result.mst);
        DEBUG_PRINT( "[DEBUG] Freed MST memory \n");
    }

    return 0;  // Add return statement
}