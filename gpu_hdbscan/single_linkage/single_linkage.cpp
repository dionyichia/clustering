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
// Recursively gather all original points under cluster `c`.
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
    int min_cluster_size)
{
    std::cout << "\n=== Running Single Linkage Clustering ===" << std::endl;

    // Make a copy of mst_edges for sorting
    std::vector<Edge> edges_copy = mst_edges;
    assert(!edges_copy.empty());
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
    std::cout << "[DEBUG] Raw smallest weight: " << smallest_weight << "\n";
    if (smallest_weight <= 0.f) {
        smallest_weight = std::numeric_limits<float>::min();
        std::cout << "[DEBUG] Adjusted smallest weight to epsilon: " << smallest_weight << "\n";
    }

    // Compute lambda_max as the inverse of smallest_mrd
    float lambda_max = 1.f / smallest_weight;
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
    for(auto &e : edges_copy){
        // if nodes are already in the same cluster
        // continue
        int c1 = find_root(e.u), c2 = find_root(e.v);
        if(c1 == c2) continue;

        // else merge them

        // lambda = 1 / MRD 
        // decrease from lambda = infinity to lambda = 0
        float lambda = 1.f / e.weight;
        
        // if clusters merged, track their death lambda
        death_lambda[c1] = death_lambda[c2] = lambda;

        // record their stability as (birth - death * size)
        stability[c1] += (birth_lambda[c1] - lambda) * sz[c1];
        stability[c2] += (birth_lambda[c2] - lambda) * sz[c2];

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

        // std::cout << "[DEBUG] Merged clusters " << c1 << " and " << c2
        //         << " into " << c_new << " at lambda=" << lambda << "\n";
    }

    // supposed to have 2N-1 clusters since there will be N-1 merges from N-1 edges
    std::cout << "[DEBUG] Total clusters created: " << next_cluster_id << "\n";

    // Finalize singleton deaths
    // Fix the finalization - add stability for ALL nodes that died
    for(int c = 0; c < next_cluster_id; ++c){
        if(death_lambda[c] > 0) {  
            // Stability already accumulated during merging
        } else if(parent[c] == c) {  // Root node - dies at lambda=0
            stability[c] += (birth_lambda[c] - 0) * sz[c];
        }
    }

    // IMPROVED CLUSTER SELECTION: Globally optimal approach
    // REPLACE your cluster selection logic with this single, clean approach:

    // Dynamic Programming table
    // Max Clusters Number Of Elements
    // Each Element Stores Total Stability of Selection and selected clusters
    std::vector<ClusterChoice> dp(max_clusters);

    std::cout << "\n=== Computing Optimal Cluster Selection ===" << std::endl;

    // Bottom-up DP: For each node, compute best selection in its subtree
    for(int c = 0; c < next_cluster_id; ++c) {

        // if cluster is less than min size, ignore it
        // stability = 0, no selected descendants
        if(sz[c] < min_cluster_size) {
            dp[c] = ClusterChoice(0.0f, {});
            continue;
        }
        
        // else, process cluster
    
        int L = left_child[c], R = right_child[c];
        
        // if no children
        if(L == -1 && R == -1) {
            // Leaf node: only choice is to select self (no children)
            dp[c] = ClusterChoice(stability[c], {c});
            std::cout << "[DEBUG] Leaf " << c << ": stability=" << stability[c] << std::endl;
        } else {
            // Internal node: compare selecting self vs optimal descendants
            ClusterChoice select_self(stability[c], {c});
            // std::cout << "Current Cluster ID" << c << std::endl;
            ClusterChoice select_descendants(0.0f, {});
            // std::cout << "[DEBUG] Left Child " << L << ": Size=" << sz[L] << std::endl;
            // Aggregate optimal solutions from children
            if(L >= 0 && sz[L] >= min_cluster_size) {
                select_descendants.total_stability += dp[L].total_stability;
                // std::cout << "[DEBUG] Selected Clusters In Left Child ";
                // for (int element : dp[L].selected_clusters) {
                //     std::cout << element << " ";
                // }
                // std::cout << std::endl;
                select_descendants.selected_clusters.insert(
                    select_descendants.selected_clusters.end(),
                    dp[L].selected_clusters.begin(),
                    dp[L].selected_clusters.end()
                );
            }
            // std::cout << "[DEBUG] Right Child " << R << ": Size=" << sz[R] << std::endl;
            if(R >= 0 && sz[R] >= min_cluster_size) {
                select_descendants.total_stability += dp[R].total_stability;
                // std::cout << "[DEBUG] Selected Clusters In Right Child ";
                // for (int element : dp[R].selected_clusters) {
                //     std::cout << element << " ";
                // }
                // std::cout << std::endl;
                select_descendants.selected_clusters.insert(
                    select_descendants.selected_clusters.end(),
                    dp[R].selected_clusters.begin(),
                    dp[R].selected_clusters.end()
                );
            }
            
            // Choose the option with higher total stability
            if(select_self.total_stability >= select_descendants.total_stability) {
                dp[c] = select_self;
                // std::cout << "[DEBUG] Node " << c << ": selecting SELF (stab=" 
                //         << select_self.total_stability << ") over descendants (stab=" 
                //         << select_descendants.total_stability << ")" << std::endl;
            } else {
                dp[c] = select_descendants;
                // std::cout << "[DEBUG] Node " << c << ": selecting DESCENDANTS (stab=" 
                //         << select_descendants.total_stability << ") over self (stab=" 
                //         << select_self.total_stability << ")" << std::endl;
            }
        }
    }

    // Extract the globally optimal solution from root(s)
    std::vector<int> final_clusters;
    std::vector<bool> is_selected(max_clusters, false);

    std::cout << "\n=== Extracting Final Clusters ===" << std::endl;

    // Find root node(s) and extract their optimal solutions
    bool found_root = false;
    for(int c = 0; c < next_cluster_id; ++c) {
        if(parent[c] == c && sz[c] >= min_cluster_size) {
            std::cout << "[DEBUG] Found root " << c << " with optimal stability=" 
                    << dp[c].total_stability << " and " << dp[c].selected_clusters.size() 
                    << " clusters" << std::endl;
            
            // Add all selected clusters from this root's optimal solution
            for(int selected : dp[c].selected_clusters) {
                if(!is_selected[selected]) {
                    is_selected[selected] = true;
                    final_clusters.push_back(selected);
                    // std::cout << "[DEBUG] Selected cluster " << selected 
                    //         << " (size=" << sz[selected] 
                    //         << ", stability=" << stability[selected] << ")" << std::endl;
                }
            }
            found_root = true;
            // break; // Assuming single root (typical case)
        }
    }

    if(!found_root) {
        std::cerr << "[ERROR] No root cluster found! Check hierarchy construction." << std::endl;
        return {};
    }

    // Final validation and statistics
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Selected " << final_clusters.size() << " clusters" << std::endl;

    float total_selected_stability = 0.0f;
    for(int c : final_clusters) {
        total_selected_stability += stability[c];
    }
    std::cout << "Total stability: " << total_selected_stability << std::endl;

    // // Validation: ensure no overlapping clusters
    // for(size_t i = 0; i < final_clusters.size(); ++i) {
    //     for(size_t j = i + 1; j < final_clusters.size(); ++j) {
    //         int c1 = final_clusters[i], c2 = final_clusters[j];
            
    //         // Function to check if c1 is ancestor of c2
    //         std::function<bool(int, int)> is_ancestor = [&](int ancestor, int descendant) -> bool {
    //             if(descendant < N_pts) return false; // Leaf nodes
    //             int L = left_child[descendant], R = right_child[descendant];
    //             if(L == ancestor || R == ancestor) return true;
    //             if(L >= N_pts && is_ancestor(ancestor, L)) return true;
    //             if(R >= N_pts && is_ancestor(ancestor, R)) return true;
    //             return false;
    //         };
            
    //         if(is_ancestor(c1, c2) || is_ancestor(c2, c1)) {
    //             std::cerr << "[ERROR] Overlapping clusters detected: " << c1 
    //                     << " and " << c2 << " have ancestor-descendant relationship!" << std::endl;
    //         }
    //     }
    // }

    // std::cout << "Cluster selection validation: " 
    //         << (final_clusters.size() > 0 ? "PASSED" : "FAILED") << std::endl;


    // Assign points to clusters
    std::vector<int> assignment(N_pts, -1);
    std::vector<std::vector<int>> clusters;
    
    for(int c : final_clusters){
        std::vector<int> mem;
        collect_members(c, N_pts, left_child, right_child, mem);
        std::vector<int> this_cluster;
        for(int p : mem){
            if(assignment[p] == -1){
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
    
    // Convert predicted clusters to label format
    std::vector<int> pred_labels(total_points, -1);  // -1 for noise/unassigned
    for (int cluster_id = 0; cluster_id < predicted_clusters.size(); ++cluster_id) {
        for (int point_id : predicted_clusters[cluster_id]) {
            if (point_id >= 0 && point_id < total_points) {
                pred_labels[point_id] = cluster_id;
            }
        }
    }
    
    // Basic statistics
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