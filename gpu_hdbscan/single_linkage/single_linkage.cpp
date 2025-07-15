#include "single_linkage.hpp"
#include "kd_tree/include/types.hpp"  // Include for Edge definition
#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>
#include <stack>
#include <hip/hip_runtime.h>

const char* clusterMethodName(clusterMethod m) {
    switch (m) {
      case clusterMethod::EOM: return "Excess Of Mass";
      case clusterMethod::Leaf:        return "Leaf";
    }
    return "Unknown";
}
__global__ void finalize_stability_kernel(
    const int* parent,
    const int* sz,
    const float* birth_lambda,
    const float* death_lambda,
    float* stability,
    int num_clusters,
    float lambda_max,
    float lambda_min
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_clusters) return;

    if (death_lambda[c] > 0) {
        // already processed
        return;
    }

    if (parent[c] == c) {
        // Root node
        // You may want to pass in num_connected_components for correct logic
        // Assuming >1 connected components here for simplicity
        float final_contribution = (lambda_max - lambda_min) * sz[c];
        atomicAdd(&stability[c], final_contribution);
    } else {
        // Singleton that was never merged
        float final_contribution = (birth_lambda[c] - lambda_min) * sz[c];
        atomicAdd(&stability[c], final_contribution);
    }
}


void parallel_finalize_stability(
    std::vector<int>& parent,
    std::vector<int>& sz,
    std::vector<float>& birth_lambda,
    std::vector<float>& death_lambda,
    std::vector<float>& stability,
    float lambda_max,
    float lambda_min
) {
    int num_clusters = parent.size();
    const size_t int_bytes = num_clusters * sizeof(int);
    const size_t float_bytes = num_clusters * sizeof(float);

    // Allocate device memory
    int *d_parent, *d_sz;
    float *d_birth_lambda, *d_death_lambda, *d_stability;

    hipMalloc(&d_parent, int_bytes);
    hipMalloc(&d_sz, int_bytes);
    hipMalloc(&d_birth_lambda, float_bytes);
    hipMalloc(&d_death_lambda, float_bytes);
    hipMalloc(&d_stability, float_bytes);

    // Copy data to device
    hipMemcpy(d_parent, parent.data(), int_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_sz, sz.data(), int_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_birth_lambda, birth_lambda.data(), float_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_death_lambda, death_lambda.data(), float_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_stability, stability.data(), float_bytes, hipMemcpyHostToDevice);

    // Kernel launch config
    int threadsPerBlock = 256;
    int blocks = (num_clusters + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    hipLaunchKernelGGL(finalize_stability_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0,
        d_parent, d_sz, d_birth_lambda, d_death_lambda, d_stability,
        num_clusters, lambda_max, lambda_min
    );

    // Wait for kernel to finish
    hipDeviceSynchronize();

    // Copy result back
    hipMemcpy(stability.data(), d_stability, float_bytes, hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_parent);
    hipFree(d_sz);
    hipFree(d_birth_lambda);
    hipFree(d_death_lambda);
    hipFree(d_stability);
}

void collect_members(int c,
                     int N_pts,
                     const std::vector<int>& left_child,
                     const std::vector<int>& right_child,
                     std::vector<int>& out)
{
    std::stack<int> stack;
    stack.push(c);
    while(!stack.empty()){
	int current = stack.top();
	stack.pop();

    if(current < N_pts){
		// Leaf Node - add to output
		out.push_back(current);
	} else {
        // Internal node - push children to stack
        // Push right child first so left child is processed first (maintains order)
		if (right_child[current] != -1) {
			stack.push(right_child[current]);
		} 
		if (left_child[current] != -1) {
			stack.push(left_child[current]);
		}
	}
	}
}

std::vector<std::vector<int>> single_linkage_clustering(
    const std::vector<Edge>& mst_edges,
    int N_pts,
    int min_cluster_size,
    clusterMethod clusterMethod
)
{
    std::cout << "\n=== Running Single Linkage Clustering ===" << std::endl;

    // ====== CLUSTER HIERARCHY TREE  ======
    // Make a copy of mst_edges for sorting
    std::vector<Edge> edges_copy = mst_edges;
    assert(!edges_copy.empty());
    
    std::sort(edges_copy.begin(), edges_copy.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
    });

    for (auto &e : edges_copy) {
    assert(e.u >= 0 && e.u < N_pts);
    assert(e.v >= 0 && e.v < N_pts);
    assert(e.weight > 0);
    }
    std::cout << "[DEBUG] Edge assertions passed.\n";

    int max_clusters = 2 * N_pts;
    std::vector<int> parent(max_clusters), sz(max_clusters);
    std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // Guard against zero-length edges
    float smallest_weight = edges_copy.front().weight;
    float largest_weight = edges_copy.back().weight;  
    std::cout << "[DEBUG] Raw smallest weight: " << smallest_weight << "\n";
    std::cout << "[DEBUG] Raw largest weight: " << largest_weight << "\n"; 
    if (smallest_weight <= 0.f) {
        smallest_weight = std::numeric_limits<float>::min();
        std::cout << "[DEBUG] Adjusted smallest weight to epsilon: " << smallest_weight << "\n";
    }

    // Compute lambda_max as the inverse of smallest_mrd
    float lambda_max = 1.f / smallest_weight;
    float lambda_min = 1.f / largest_weight;  
    
    std::cout << "[DEBUG] lambda_max: " << lambda_max << "\n";
    
    /* initialise all points as singleton clusters */
    for(int i = 0; i < N_pts; ++i){
        parent[i] = i;
        sz[i] = 1;
        birth_lambda[i] = lambda_max;
        death_lambda[i] = 0;
        stability[i] = 0;
    }
    int next_cluster_id = N_pts;

    // initialise each point as its own singleton cluster 
    std::cout << "[DEBUG] Initialized " << next_cluster_id << " singleton clusters\n";

    // lambda to find root (path-compressed):
    auto find_root = [&](int x){
        int root = x;
        while(parent[root] != root) root = parent[root];
        while(parent[x] != root){
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    // Build hierarchy
    std::cout << "\n=== HIERARCHY BUILDING DEBUG ===" << std::endl;
    for(int edge_idx = 0; edge_idx < edges_copy.size(); ++edge_idx) {
        auto &e = edges_copy[edge_idx];
        
        // if nodes are already in the same cluster, continue
        int c1 = find_root(e.u), c2 = find_root(e.v);
        if(c1 == c2) continue;

        // lambda = 1 / MRD - decrease from lambda = infinity to lambda = 0
        float lambda = 1.f / e.weight;
    
        // Calculate stability contributions before merging
        float stability_c1 = (birth_lambda[c1] - lambda) * sz[c1];
        float stability_c2 = (birth_lambda[c2] - lambda) * sz[c2];
        
        // Set death lambda and record stability
        death_lambda[c1] = lambda;
        death_lambda[c2] = lambda;
        stability[c1] += stability_c1;
        stability[c2] += stability_c2;
        // make new cluster
        int c_new = next_cluster_id++;
        parent[c1] = parent[c2] = c_new;
        parent[c_new] = c_new;
        sz[c_new]           = sz[c1] + sz[c2];
        birth_lambda[c_new] = lambda;
        stability[c_new]    = 0;
        death_lambda[c_new] = 0;
        left_child[c_new]   = c1;
        right_child[c_new]  = c2;
    }


    // supposed to have 2N-1 clusters since there will be N-1 merges from N-1 edges
    std::cout << "[DEBUG] Total clusters created: " << next_cluster_id << "\n";

    // ====== CONDENSE TREE ======
    // IDENTIFY ROOT CLUSTERS AND CONNECTED COMPONENTS
    std::vector<int> root_clusters;
    for(int c = 0; c < next_cluster_id; ++c) {
        if(parent[c] == c) {  // This is a root cluster
            root_clusters.push_back(c);
        }
    }

    int num_connected_components = root_clusters.size();
    std::cout << "[DEBUG] Found " << num_connected_components << " connected components (roots): ";
    for(int root : root_clusters) {
        std::cout << root << " ";
    }
    std::cout << std::endl;

    // Finalize singleton deaths
    std::cout << "\n=== FINALIZING STABILITY DEBUG ===" << std::endl;
    for(int c = 0; c < next_cluster_id; ++c){
        if(death_lambda[c] > 0) {  
            // Already processed during merging
            if(c < 20) {
                std::cout << "Cluster " << c << " (merged): final stability=" << stability[c] << std::endl;
            }
        } else if(parent[c] == c) {  // Root node - dies at lambda_min 
            if(num_connected_components == 1) {
                stability[c] = 0;  // Single component spanning whole dataset
            } else {
                // Multiple roots = real connected components
                // std::cout << "Cluster " << c << " birth_lambda:" << birth_lambda[c] 
                //         << " lambda_min:" << lambda_min << std::endl;
                float final_contribution = (birth_lambda[c] - lambda_min) * sz[c];
                stability[c] += final_contribution;
            }
        } else {
            // Singleton that was never merged
            float final_contribution = (birth_lambda[c] - lambda_min) * sz[c]; 
            stability[c] += final_contribution;
        }
    }
    // parallel_finalize_stability(
    //     parent,
    //     sz,
    //     birth_lambda,
    //     death_lambda,
    //     stability,
    //     lambda_max,
    //     lambda_min
    // );
    std::vector<int> final_clusters;

    // ====== CLUSTER SELECTION ======
    switch(clusterMethod){
        case clusterMethod::EOM: {
        // === EXCESS OF MASS CLUSTER SELECTION===

        std::cout << "\n=== Computing Optimal Cluster Selection ===" << std::endl;

        // Track which clusters are selected in the optimal solution
        std::vector<bool> is_selected(max_clusters, false);
        
        // Bottom-up cluster selection: for each node, decide independently
        std::function<void(int)> select_clusters = [&](int node) {
            if(node < 0) return;
            
            int L = left_child[node], R = right_child[node];
            
            // if no children and sz[node] >= min_cluster_size
            // select node
            if(L == -1 && R == -1) {
                if(sz[node] >= min_cluster_size) {
                    is_selected[node] = true;
                    std::cout << "[DEBUG] Leaf cluster " << node 
                            << " selected (size=" << sz[node] 
                            << ", stability=" << stability[node] << ")" << std::endl;
                } else {
                    std::cout << "[DEBUG] Leaf cluster " << node 
                            << " too small (size=" << sz[node] 
                            << "), will contribute to parent" << std::endl;
                }
                return;
            }

            // Leaf nodes: select if they meet minimum size, otherwise they contribute to parent
            // Recursively solve left and right subtree of node
            // This tells us which clusters are selected from left and right subtree
            select_clusters(L);
            select_clusters(R);
            
            // Calculate stability of optimal selection from children
            float children_stability = 0.0f;
            std::vector<int> children_to_unselect;
            
            // Helper function to collect stability from a subtree
            std::function<float(int)> get_subtree_stability = [&](int subtree_root) -> float {
                if(subtree_root < 0) return 0.0f;
                
                float total = 0.0f;
                // if the subtree root is selected, its stability is higher than its children
                // so children's stability not relevant 
                if(is_selected[subtree_root]) {
                    total += stability[subtree_root];
                }
                
                // If this node wasn't selected, add its children's contributions
                if(!is_selected[subtree_root]) {
                    total += get_subtree_stability(left_child[subtree_root]);
                    total += get_subtree_stability(right_child[subtree_root]);
                }
                
                return total;
            };
            
            // For each child, get the optimal stability from its subtree
            if(L >= 0) {
                children_stability += get_subtree_stability(L);
            }
            if(R >= 0) {
                children_stability += get_subtree_stability(R);
            }
            
            // Collect all nodes that would need to be unselected if we select this node
            std::function<void(int)> collect_selected_nodes = [&](int subtree_root) {
                if(subtree_root < 0) return;
                if(is_selected[subtree_root]) {
                    children_to_unselect.push_back(subtree_root);
                } else {
                    collect_selected_nodes(left_child[subtree_root]);
                    collect_selected_nodes(right_child[subtree_root]);
                }
            };
            
            collect_selected_nodes(L);
            collect_selected_nodes(R);
            
            // Compare: this node vs optimal children selection
            if(sz[node] >= min_cluster_size && stability[node] > children_stability) {
                // Unselect all descendants, select this node
                for(int child : children_to_unselect) {
                    is_selected[child] = false;
                }
                is_selected[node] = true;
                
                std::cout << "[DEBUG] Node " << node << " selected over descendants" 
                        << " (node_stability=" << stability[node] 
                        << " vs children_stability=" << children_stability << ")" << std::endl;
            } else if(sz[node] >= min_cluster_size) {
                std::cout << "[DEBUG] Node " << node << " NOT selected, keeping descendants" 
                        << " (node_stability=" << stability[node] 
                        << " vs children_stability=" << children_stability << ")" << std::endl;
            } else {
                std::cout << "[DEBUG] Node " << node << " too small (size=" << sz[node] 
                        << "), stability will contribute to parent" << std::endl;
            }
        };

        // Process all trees (handle multiple roots)
        for(int c = 0; c < next_cluster_id; ++c) {
            if(parent[c] == c) {  // Root node
                std::cout << "[DEBUG] Processing tree rooted at " << c << std::endl;
                select_clusters(c);
            }
        }

        std::cout << "\n=== Extracting Final Clusters ===" << std::endl;
        
        // Build final clusters list from selected nodes
        float total_selected_stability = 0.0f;
        for(int c = 0; c < next_cluster_id; ++c) {
            if(is_selected[c]) {
                final_clusters.push_back(c);
                total_selected_stability += stability[c];
                std::cout << "[DEBUG] Final cluster " << c 
                        << " (size=" << sz[c] 
                        << ", stability=" << stability[c] << ")" << std::endl;
            }
        }
        
        // Final validation and statistics
        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Selected " << final_clusters.size() << " clusters" << std::endl;
        std::cout << "Total stability: " << total_selected_stability << std::endl;
        break;
    }
    case clusterMethod::Leaf: {
        // === LEAF CLUSTER SELECTION ===
        final_clusters.reserve(N_pts);
        std::stack<int> stk;

        // 1) Push each root (top‐level cluster) onto the stack
        for (int c = 0; c < next_cluster_id; ++c) {
            if (parent[c] == c && sz[c] >= min_cluster_size) {
                stk.push(c);
            }
        }

        // 2) DFS: pop a node, decide if it's a leaf or push children
        while (!stk.empty()) {
            int c = stk.top();
            stk.pop();

            // skip if too small
            if (sz[c] < min_cluster_size) continue;

            int L = left_child[c], R = right_child[c];
            bool left_valid  = (L >= 0 && sz[L] >= min_cluster_size);
            bool right_valid = (R >= 0 && sz[R] >= min_cluster_size);

            // if neither child is a valid cluster, c is a “leaf”
            if (!left_valid && !right_valid) {
                final_clusters.push_back(c);
            }
            else {
                // otherwise, descend into any valid child
                if (left_valid)  stk.push(L);
                if (right_valid) stk.push(R);
            }
        }
        break;
    }
    }
    // ==== LABELLING ====
    // Assign points to clusters
    std::vector<int> assignment(N_pts, -1);
    std::vector<std::vector<int>> clusters;
    
    for(int c : final_clusters){
        std::vector<int> mem;
        collect_members(c, N_pts, left_child, right_child, mem);
        std::vector<int> this_cluster;
        for(int p : mem){
            if(assignment[p] == -1) {
                assignment[p] = clusters.size();
                this_cluster.push_back(p);
            }
        }
        if(!this_cluster.empty()){
            clusters.push_back(std::move(this_cluster));
            std::cout << "[DEBUG] Cluster " << (clusters.size()-1)
                    << " got " << clusters.back().size() << " points\n";
        }
    }
    return clusters;
}

// Helper function to calculate entropy
double calculateEntropy(const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int label : labels) {
        counts[label]++;
    }
    
    double entropy = 0.0;
    int total = labels.size();
    for (const auto& pair : counts) {
        if (pair.second > 0) {
            double p = static_cast<double>(pair.second) / total;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// Helper function to calculate mutual information
double calculateMutualInformation(const std::vector<int>& true_labels, 
                                const std::vector<int>& pred_labels) {
    std::map<std::pair<int, int>, int> joint_counts;
    std::map<int, int> true_counts, pred_counts;
    
    int n = true_labels.size();
    for (int i = 0; i < n; ++i) {
        joint_counts[{true_labels[i], pred_labels[i]}]++;
        true_counts[true_labels[i]]++;
        pred_counts[pred_labels[i]]++;
    }
    
    double mi = 0.0;
    for (const auto& joint_pair : joint_counts) {
        int true_label = joint_pair.first.first;
        int pred_label = joint_pair.first.second;
        int joint_count = joint_pair.second;
        
        double p_xy = static_cast<double>(joint_count) / n;
        double p_x = static_cast<double>(true_counts[true_label]) / n;
        double p_y = static_cast<double>(pred_counts[pred_label]) / n;
        
        if (p_xy > 0 && p_x > 0 && p_y > 0) {
            mi += p_xy * std::log2(p_xy / (p_x * p_y));
        }
    }
    return mi;
}

// Calculate Adjusted Rand Index
double calculateAdjustedRandIndex(const std::vector<int>& true_labels, 
                                const std::vector<int>& pred_labels) {
    int n = true_labels.size();
    std::map<std::pair<int, int>, int> contingency;
    std::map<int, int> true_counts, pred_counts;
    
    // Build contingency table
    for (int i = 0; i < n; ++i) {
        contingency[{true_labels[i], pred_labels[i]}]++;
        true_counts[true_labels[i]]++;
        pred_counts[pred_labels[i]]++;
    }
    
    // Calculate components for ARI formula
    double sum_comb_c = 0.0;  // sum of C(n_ij, 2)
    for (const auto& pair : contingency) {
        int count = pair.second;
        if (count >= 2) {
            sum_comb_c += static_cast<double>(count * (count - 1)) / 2.0;
        }
    }
    
    double sum_comb_a = 0.0;  // sum of C(a_i, 2)
    for (const auto& pair : true_counts) {
        int count = pair.second;
        if (count >= 2) {
            sum_comb_a += static_cast<double>(count * (count - 1)) / 2.0;
        }
    }
    
    double sum_comb_b = 0.0;  // sum of C(b_j, 2)
    for (const auto& pair : pred_counts) {
        int count = pair.second;
        if (count >= 2) {
            sum_comb_b += static_cast<double>(count * (count - 1)) / 2.0;
        }
    }
    
    double total_comb = static_cast<double>(n * (n - 1)) / 2.0;
    
    double expected_index = (sum_comb_a * sum_comb_b) / total_comb;
    double max_index = (sum_comb_a + sum_comb_b) / 2.0;
    
    if (max_index - expected_index == 0) {
        return 0.0;  // Perfect agreement or no agreement possible
    }
    
    return (sum_comb_c - expected_index) / (max_index - expected_index);
}

// Calculate best matching accuracy using Hungarian algorithm approximation
double calculateBestMatchingAccuracy(const std::vector<int>& true_labels, 
                                   const std::vector<int>& pred_labels) {
    // Build confusion matrix
    std::set<int> true_set(true_labels.begin(), true_labels.end());
    std::set<int> pred_set(pred_labels.begin(), pred_labels.end());
    
    std::map<int, std::map<int, int>> confusion;
    for (int i = 0; i < true_labels.size(); ++i) {
        confusion[true_labels[i]][pred_labels[i]]++;
    }
    
    // Greedy assignment (simple approximation of Hungarian algorithm)
    std::set<int> used_pred;
    int correct = 0;
    
    // For each true cluster, find the best matching predicted cluster
    for (int true_cluster : true_set) {
        int best_pred = -1;
        int best_count = 0;
        
        for (const auto& pred_pair : confusion[true_cluster]) {
            int pred_cluster = pred_pair.first;
            int count = pred_pair.second;
            
            if (used_pred.find(pred_cluster) == used_pred.end() && count > best_count) {
                best_count = count;
                best_pred = pred_cluster;
            }
        }
        
        if (best_pred != -1) {
            used_pred.insert(best_pred);
            correct += best_count;
        }
    }
    
    return static_cast<double>(correct) / true_labels.size();
}

/**
 * Evaluate clustering results against ground truth labels
 * @param true_labels: Ground truth cluster labels from CSV
 * @param predicted_clusters: Clusters generated by single linkage algorithm
 * @param total_points: Total number of data points
 * @return ClusterMetrics structure with various evaluation metrics
 */
ClusterMetrics evaluateClustering(const std::vector<int>& true_labels,
                                const std::vector<std::vector<int>>& predicted_clusters,
                                int total_points) {
    
    ClusterMetrics metrics;
    int noise_points = 0;
    // Convert predicted clusters to label format
    std::vector<int> pred_labels(total_points, -1);  // -1 for noise/unassigned
    for (int cluster_id = 0; cluster_id < predicted_clusters.size(); ++cluster_id) {
        for (int point_id : predicted_clusters[cluster_id]) {
            if (point_id >= 0 && point_id < total_points) {
                pred_labels[point_id] = cluster_id;
            }
        }
    }

    for (int i: pred_labels){
        if (i == -1){
            noise_points += 1;
        }
    }
    // Basic statistics
    metrics.noise_points = noise_points;
    metrics.total_points = total_points;
    metrics.num_predicted_clusters = predicted_clusters.size();
    std::set<int> unique_true(true_labels.begin(), true_labels.end());
    metrics.num_true_clusters = unique_true.size();
    
    // Calculate entropies
    double h_true = calculateEntropy(true_labels);
    double h_pred = calculateEntropy(pred_labels);
    
    // Calculate mutual information
    double mi = calculateMutualInformation(true_labels, pred_labels);
    
    // Normalized Mutual Information
    if (h_true + h_pred > 0) {
        metrics.normalized_mutual_info = 2.0 * mi / (h_true + h_pred);
    } else {
        metrics.normalized_mutual_info = 0.0;
    }
    
    // Homogeneity and Completeness
    if (h_pred > 0) {
        metrics.homogeneity = mi / h_pred;
    } else {
        metrics.homogeneity = 1.0;  // Perfect homogeneity if only one cluster
    }
    
    if (h_true > 0) {
        metrics.completeness = mi / h_true;
    } else {
        metrics.completeness = 1.0;  // Perfect completeness if only one true cluster
    }
    
    // V-measure (harmonic mean of homogeneity and completeness)
    if (metrics.homogeneity + metrics.completeness > 0) {
        metrics.v_measure = 2.0 * (metrics.homogeneity * metrics.completeness) / 
                           (metrics.homogeneity + metrics.completeness);
    } else {
        metrics.v_measure = 0.0;
    }
    
    // Adjusted Rand Index
    metrics.adjusted_rand_index = calculateAdjustedRandIndex(true_labels, pred_labels);
    
    // Best matching accuracy
    metrics.accuracy = calculateBestMatchingAccuracy(true_labels, pred_labels);
    
    return metrics;
}

/**
 * Print clustering evaluation results
 */
void printClusteringEvaluation(const ClusterMetrics& metrics, bool quiet_mode) {
    if (quiet_mode) {
        // Output only essential metrics for parsing
        std::cout << "EVAL_METRICS:"
                  << " ARI=" << metrics.adjusted_rand_index
                  << " NMI=" << metrics.normalized_mutual_info
                  << " ACC=" << metrics.accuracy
                  << " V_MEASURE=" << metrics.v_measure
                  << std::endl;
    } else {
        std::cout << "\n=== Clustering Evaluation Results ===" << std::endl;
        std::cout << "Total points: " << metrics.total_points << std::endl;
        std::cout << "Noise points: " << metrics.noise_points << std::endl;
        std::cout << "True clusters: " << metrics.num_true_clusters << std::endl;
        std::cout << "Predicted clusters: " << metrics.num_predicted_clusters << std::endl;
        std::cout << "\nMetrics:" << std::endl;
        std::cout << "  Adjusted Rand Index: " << std::fixed << std::setprecision(4) 
                  << metrics.adjusted_rand_index << std::endl;
        std::cout << "  Normalized Mutual Info: " << std::fixed << std::setprecision(4) 
                  << metrics.normalized_mutual_info << std::endl;
        std::cout << "  Homogeneity: " << std::fixed << std::setprecision(4) 
                  << metrics.homogeneity << std::endl;
        std::cout << "  Completeness: " << std::fixed << std::setprecision(4) 
                  << metrics.completeness << std::endl;
        std::cout << "  V-measure: " << std::fixed << std::setprecision(4) 
                  << metrics.v_measure << std::endl;
        std::cout << "  Best Matching Accuracy: " << std::fixed << std::setprecision(4) 
                  << metrics.accuracy << std::endl;
        std::cout << "===================================" << std::endl;
    }
}