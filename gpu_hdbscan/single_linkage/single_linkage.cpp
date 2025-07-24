#include "single_linkage.hpp"
#include "kd_tree/include/types.hpp"  // Include for Edge definition
#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>
#include <vector>
#include <queue>
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

// Function to perform BFS traversal from a given node
std::vector<int> bfs_from_node(int start_node, const std::vector<int>& left_child, 
                               const std::vector<int>& right_child, int n_samples) {
    std::vector<int> result;
    std::queue<int> queue;
    queue.push(start_node);
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // If it's an internal node (not a leaf/sample)
        if (current >= n_samples) {
            if (left_child[current] != -1) {
                queue.push(left_child[current]);
            }
            if (right_child[current] != -1) {
                queue.push(right_child[current]);
            }
        }
    }
    
    return result;
}

// Constructs Condensed Tree from Merge Hierarchy
// In Merge Hierarchy, birth lambda of parent is death lambda of children (since merging from bottom up)
// However in Condensed Tree, each Condensed Node stores the following info (parent,child,lambda,child_size)
// It tells us "At lambda = λ, cluster [parent] broke into cluster [child] and its size is child_size(child_size ≥ min_cluster_size).”
// CondensedNode will only store clusters >= min_cluster_size (valid clusters) or 1 (singleton clusters)
std::vector<CondensedNode> condense_tree(const std::vector<int>& left_child,
                                        const std::vector<int>& right_child,
                                        const std::vector<float>& birth_lambda,
                                        const std::vector<float>& death_lambda,
                                        const std::vector<int>& sz,
                                        int n_samples,
                                        int root_node,
                                        int& next_label,
                                        int min_cluster_size) {
    
    std::vector<CondensedNode> condensed_tree;
    // left_child.size() is arbitrary, can use size() of any array which has size == number of nodes
    std::vector<int> relabel(left_child.size(), -1);
    std::vector<bool> ignore(left_child.size(), false);
    
    // Get nodes in BFS order from root
    std::vector<int> node_list = bfs_from_node(root_node, left_child, right_child, n_samples);
    
    // Initialize root relabeling with global counter
    if (next_label == 0) {
        next_label = n_samples;
    }
    relabel[root_node] = next_label++;
    
 //   std::cout << "[DEBUG] Starting condense tree with min_cluster_size: " << min_cluster_size << "\n";
  //  std::cout << "[DEBUG] Root node: " << root_node << ", relabeled to: " << relabel[root_node] << "\n";
    
    // Process nodes in BFS order
    for (int node : node_list) {
        // Skip if already processed or if it's a singleton cluster 
        if (ignore[node] || node < n_samples) {
            continue;
        }
        
        int left = left_child[node];
        int right = right_child[node];
        
        // Skip if no children
        if (left == -1 || right == -1) {
            continue;
        }
        
        // Get lambda value (birth lambda of children = death lambda of parent)
        float lambda_value = death_lambda[node];
        
        // Get cluster sizes
        int left_count = (left >= n_samples) ? sz[left] : 1;
        int right_count = (right >= n_samples) ? sz[right] : 1;
        
    //    std::cout << "[DEBUG] Processing node " << node << " (relabeled: " << relabel[node] 
      //          << ") with children " << left << "(" << left_count << ") and " 
       //         << right << "(" << right_count << ")\n";
        
        // Case 1: Both children are large enough clusters
        if (left_count >= min_cluster_size && right_count >= min_cluster_size) {
            // Create entries for both children
            relabel[left] = next_label++;
            condensed_tree.emplace_back(relabel[node], relabel[left], lambda_value, left_count);
            
            relabel[right] = next_label++;
            condensed_tree.emplace_back(relabel[node], relabel[right], lambda_value, right_count);
            
        //    std::cout << "[DEBUG] Both children large enough - created condensed nodes\n";
        }
        // Case 2: Both children are too small
        else if (left_count < min_cluster_size && right_count < min_cluster_size) {
            // Get all leaves in invalid cluster 
            auto left_leaves = bfs_from_node(left, left_child, right_child, n_samples);
            for (int leaf : left_leaves) {
                // Check if its a valid singleton cluster
                if (leaf < n_samples) {  
                    // Track in Condensed Tree as points falling out of valid parent cluster as noise
                    condensed_tree.emplace_back(relabel[node], leaf, lambda_value, 1);
                }
                ignore[leaf] = true;
            }
            // Get all leaves in invalid cluster 
            auto right_leaves = bfs_from_node(right, left_child, right_child, n_samples);
            for (int leaf : right_leaves) {
                // Check if its a valid singleton cluster
                if (leaf < n_samples) {
                    // Track in Condensed Tree as points falling out of a valid cluster as noise  
                    condensed_tree.emplace_back(relabel[node], leaf, lambda_value, 1);
                }
                ignore[leaf] = true;
            }
            
          //  std::cout << "[DEBUG] Both children too small - added leaf nodes directly\n";
        }
        // if only one child is big enough, don't make a new cluster
        // it would just be the parent cluster shrinking into a smaller cluster 
        // Case 3: Left child too small, right child large enough
        else if (left_count < min_cluster_size) {
            // Right child inherits parent's label
            relabel[right] = relabel[node];
            
            // Get all leaves in invalid cluster 
            auto left_leaves = bfs_from_node(left, left_child, right_child, n_samples);
            for (int leaf : left_leaves) {
                // Check if its a valid singleton cluster
                if (leaf < n_samples) { 
                    // Track in Condensed Tree as points falling out of a valid cluster as noise  
                    condensed_tree.emplace_back(relabel[node], leaf, lambda_value, 1);
                }
                ignore[leaf] = true;
            }
            
     //       std::cout << "[DEBUG] Left child too small - right inherits parent label\n";
        }
        // Case 4: Right child too small, left child large enough
        else {
            // Left child inherits parent's label
            relabel[left] = relabel[node];
            
            // Get all leaves in invalid cluster 
            auto right_leaves = bfs_from_node(right, left_child, right_child, n_samples);
            for (int leaf : right_leaves) {
                // Check if its a valid singleton cluster
                if (leaf < n_samples) {
                    // Track in Condensed Tree as points falling out of a valid cluster as noise   
                    condensed_tree.emplace_back(relabel[node], leaf, lambda_value, 1);
                }
                ignore[leaf] = true;
            }
            
     //       std::cout << "[DEBUG] Right child too small - left inherits parent label\n";
        }
    }
    
   // std::cout << "[DEBUG] Condensed tree has " << condensed_tree.size() << " nodes\n";
    
    return condensed_tree;
}


// Calculate stability for each cluster 
std::map<int, ClusterStability> calculate_cluster_stability(const std::vector<CondensedNode>& condensed_tree) 
{
    // Dictionary storing clusters as {ClusterID, ClusterStability}
    std::map<int, ClusterStability> cluster_stability;
    
    // Initialize all clusters
    for (const auto& node : condensed_tree) {
        // if parent cluster is not in dictionary, add it
        if (cluster_stability.find(node.parent) == cluster_stability.end()) {
            cluster_stability[node.parent] = ClusterStability();
        }
        // if cluster is bigger than 1, it is a valid subcluster 
        // condensed tree only stores singleton cluster or clusters with size >= min_cluster_size
        if (node.cluster_size > 1) {
            // if child cluster is not in dictionary, add it
            if (cluster_stability.find(node.child) == cluster_stability.end()) {
                cluster_stability[node.child] = ClusterStability();
            }
            cluster_stability[node.parent].children.push_back(node.child);
            cluster_stability[node.child].cluster_size = node.cluster_size;
        }
        // else if cluster size is 1, it is a point falling out of the cluster 
    }
    
    // Calculate stability of parent clusters
    for (const auto& node : condensed_tree) {
        // All nodes in the condensed tree contribute to parent stability
        // The contribution is lambda_val * cluster_size
        cluster_stability[node.parent].stability += node.lambda_val * node.cluster_size;
    }
    
    return cluster_stability;
}

// Function to calculate cluster selection epsilon using elbow method
// Finds the first stabilization point in bottom-up traversal
float calculate_cluster_selection_epsilon(const std::vector<CondensedNode>& condensed_tree,
                                        const std::map<int, ClusterStability>& cluster_stability) {
    
    if (condensed_tree.empty()) {
        std::cout << "[WARNING] Empty condensed tree, returning default epsilon\n";
        return 0.0f;
    }
    
    // Step 1: Collect all unique lambda values and sort them (bottom-up)
    std::set<float> lambda_set;
    for (const auto& node : condensed_tree) {
        if (node.cluster_size != 1) {
            lambda_set.insert(node.lambda_val);
        }
    }
    
    std::vector<float> lambda_values(lambda_set.begin(), lambda_set.end());
    std::sort(lambda_values.begin(), lambda_values.end()); // ascending order (bottom-up)
    
    if (lambda_values.size() < 3) {
        std::cout << "[WARNING] Not enough lambda values for elbow detection, using median\n";
        return lambda_values[lambda_values.size() / 2];
    }
    
    // Step 2: Calculate cluster metrics at each lambda level
    std::vector<float> cluster_counts;
    
    for (float lambda : lambda_values) {
        // Count active clusters at this lambda level
        std::set<int> active_clusters;
        
        for (const auto& node : condensed_tree) {
            if (node.lambda_val <= lambda) {
                // This merge has happened, so child clusters exist
                if (node.cluster_size > 1) {
                    active_clusters.insert(node.child);
                }
            }
        }
        
        cluster_counts.push_back(static_cast<float>(active_clusters.size()));
        
        // std::cout << "[DEBUG] Lambda: " << lambda 
        //           << ", Active clusters: " << active_clusters.size() 
        //           << ", Cumulative stability: " << total_stability << "\n";
    }
    
    // Step 3: Calculate rate of change (first derivative)
    std::vector<float> rate_of_change;
    for (size_t i = 1; i < cluster_counts.size(); ++i) {
        float delta_clusters = cluster_counts[i] - cluster_counts[i-1];
        float delta_lambda = lambda_values[i] - lambda_values[i-1];
        
        if (delta_lambda > 0) {
            rate_of_change.push_back(delta_clusters / delta_lambda);
        } else {
            rate_of_change.push_back(0.0f);
        }
    }
    
    // Step 4: Find elbow point using second derivative
    std::vector<float> second_derivative;
    for (size_t i = 1; i < rate_of_change.size(); ++i) {
        float delta_rate = rate_of_change[i] - rate_of_change[i-1];
        float delta_lambda = lambda_values[i+1] - lambda_values[i];
        
        if (delta_lambda > 0) {
            second_derivative.push_back(delta_rate / delta_lambda);
        } else {
            second_derivative.push_back(0.0f);
        }
    }
    
    // Step 5: Find first stabilization point (first elbow from bottom-up)
    // Look for the first point where the second derivative approaches zero
    // after initial rapid change, indicating stabilization of small clusters
    
    float epsilon = lambda_values[0]; // default to smallest lambda
    float min_second_deriv_threshold = 0.1f; // threshold for stabilization
    bool found_initial_change = false;
    
    for (size_t i = 0; i < second_derivative.size(); ++i) {
        float abs_second_deriv = std::abs(second_derivative[i]);
        
        std::cout << "[DEBUG] Index: " << i+2 
                  << ", Lambda: " << lambda_values[i+2]
                  << ", 2nd derivative: " << second_derivative[i] << "\n";
        
        // First, detect if we've seen significant change
        if (!found_initial_change && abs_second_deriv > min_second_deriv_threshold) {
            found_initial_change = true;
            continue;
        }
        
        // Then look for stabilization after initial change
        if (found_initial_change && abs_second_deriv < min_second_deriv_threshold) {
            epsilon = lambda_values[i+2]; // +2 because second derivative is offset by 2
            std::cout << "[INFO] Elbow point detected at lambda: " << epsilon << "\n";
            break;
        }
    }
    
    // Alternative approach: If no clear elbow, use cluster count stabilization
    // if (epsilon == lambda_values[0] && lambda_values.size() > 5) {
    //     std::cout << "[INFO] No clear elbow found, using cluster count stabilization\n";
        
    //     // Find where cluster count stops changing rapidly
    //     for (size_t i = 2; i < cluster_counts.size() - 1; ++i) {
    //         float prev_change = std::abs(cluster_counts[i] - cluster_counts[i-1]);
    //         float next_change = std::abs(cluster_counts[i+1] - cluster_counts[i]);
            
    //         // If change is small and consistent, we've found stabilization
    //         if (prev_change <= 1.0f && next_change <= 1.0f) {
    //             epsilon = lambda_values[i];
    //             std::cout << "[INFO] Cluster stabilization point at lambda: " << epsilon << "\n";
    //             break;
    //         }
    //     }
    // }
    
    // // Final fallback: use median if still no good epsilon found
    // if (epsilon == lambda_values[0]) {
    //     epsilon = lambda_values[lambda_values.size() / 3]; // Use lower third instead of median
    //     std::cout << "[INFO] Using fallback epsilon (lower third): " << epsilon << "\n";
    // }
    
    std::cout << "[RESULT] Selected cluster_selection_epsilon: " << epsilon << "\n";
    return epsilon;
}

// BFS to find all descendants of a cluster
std::vector<int> bfs_descendants(int cluster_id, 
                                const std::map<int, ClusterStability>& cluster_stability) {
    std::vector<int> descendants;
    std::queue<int> queue;
    queue.push(cluster_id);
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        descendants.push_back(current);
        
        auto it = cluster_stability.find(current);
        if (it != cluster_stability.end()) {
            for (int child : it->second.children) {
                queue.push(child);
            }
        }
    }
    
    return descendants;
}


// Excess of Mass cluster selection (like sklearn)
std::set<int> excess_of_mass_selection(std::map<int, ClusterStability>& cluster_stability,
                                    int max_cluster_size) {
    
    // Get clusters in reverse topological order (largest IDs first)
    std::vector<int> node_list;
    for (const auto& [cluster_id, _] : cluster_stability) {
        node_list.push_back(cluster_id);
    }
    std::sort(node_list.rbegin(), node_list.rend());
    
    std::cout << "[DEBUG] EOM selection processing " << node_list.size() << " clusters\n";
    
    // Apply Excess of Mass algorithm
    for (int node : node_list) {
        // Calculate subtree stability (sum of children's stabilities)
        float subtree_stability = 0.0f;
        for (int child : cluster_stability[node].children) {
            subtree_stability += cluster_stability[child].stability;
        }
        
       // std::cout << "[DEBUG] Cluster " << node << ": own_stability=" 
       //           << cluster_stability[node].stability << ", subtree_stability=" 
       //           << subtree_stability << ", size=" << cluster_stability[node].cluster_size << "\n";
        
        // EOM decision: keep cluster if its stability > sum of children's stability
        // Check parent cluster doesn't exceed max_cluster_size
        if (subtree_stability > cluster_stability[node].stability || 
            cluster_stability[node].cluster_size > max_cluster_size) {
            
            // Select children instead of this cluster
            cluster_stability[node].is_cluster = false;
            cluster_stability[node].stability = subtree_stability;
            
        //    std::cout << "[DEBUG] Cluster " << node << " rejected - selecting children\n";
        } else {
            // Keep this cluster, mark all descendants as non-clusters
            std::vector<int> descendants = bfs_descendants(node, cluster_stability);
            for (int desc : descendants) {
                if (desc != node) {
                    cluster_stability[desc].is_cluster = false;
                }
            }
        //    std::cout << "[DEBUG] Cluster " << node << " selected - " 
        //              << descendants.size() << " descendants marked as non-clusters\n";
        }
    }
    
    // Collect selected clusters
    std::set<int> selected_clusters;
    for (const auto& [cluster_id, info] : cluster_stability) {
        if (info.is_cluster) {
            selected_clusters.insert(cluster_id);
            std::cout << "[DEBUG] Final selected cluster: " << cluster_id 
                      << " (stability=" << info.stability << ")\n";
        }
    }
    
    return selected_clusters;
}

// Function to get leaf nodes from the cluster tree
std::set<int> get_cluster_tree_leaves(const std::vector<CondensedNode>& condensed_tree) {
    std::set<int> all_parents;
    std::set<int> all_children;
    
    // Collect all parent and child cluster IDs
    for (const auto& node : condensed_tree) {
        if (node.cluster_size > 1) {  // Only consider cluster nodes, not point nodes
            all_parents.insert(node.parent);
            all_children.insert(node.child);
        }
    }
    
    // Leaves are children that are not parents
    std::set<int> leaves;
    for (int child : all_children) {
        if (all_parents.find(child) == all_parents.end()) {
            leaves.insert(child);
        }
    }
    
    return leaves;
}

// Function to find minimum parent ID (root cluster)
int find_min_parent(const std::vector<CondensedNode>& condensed_tree) {
    int min_parent = std::numeric_limits<int>::max();
    for (const auto& node : condensed_tree) {
        if (node.cluster_size > 1) {  // Only consider cluster nodes
            min_parent = std::min(min_parent, node.parent);
        }
    }
    return min_parent;
}

// Helper function to traverse upwards in the cluster tree
int traverse_upwards(const std::vector<CondensedNode>& cluster_tree,
                    double cluster_selection_epsilon,
                    int cluster_id,
                    bool allow_single_cluster) {
    int current_cluster = cluster_id;
    
    while (true) {
        // Find the parent of the current cluster
        int parent = -1;
        for (const auto& node : cluster_tree) {
            if (node.child == current_cluster) {
                parent = node.parent;
                break;
            }
        }
        
        // If no parent found, we've reached the root
        if (parent == -1) {
            return current_cluster;
        }
        
        // Find the birth epsilon (1/value) of the parent
        double parent_birth_epsilon = 0.0;
        for (const auto& node : cluster_tree) {
            if (node.child == parent) {
                parent_birth_epsilon = 1.0 / node.lambda_val;
                break;
            }
        }
        
        // If parent has sufficient epsilon, return it
        if (parent_birth_epsilon >= cluster_selection_epsilon) {
            return parent;
        }
        
        // Continue traversing upwards
        current_cluster = parent;
    }
}

// Helper function to perform BFS from a cluster to find all descendant nodes
std::vector<int> bfs_from_cluster_tree(const std::vector<CondensedNode>& cluster_tree,
                                      int root_cluster) {
    std::vector<int> result;
    std::queue<int> queue;
    std::set<int> visited;
    
    queue.push(root_cluster);
    visited.insert(root_cluster);
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // Find all children of current cluster
        for (const auto& node : cluster_tree) {
            if (node.parent == current && visited.find(node.child) == visited.end()) {
                queue.push(node.child);
                visited.insert(node.child);
            }
        }
    }
    
    return result;
}


// Main epsilon search function
std::set<int> epsilon_search(const std::set<int>& leaves,
                           const std::vector<CondensedNode>& cluster_tree,
                           double cluster_selection_epsilon,
                           bool allow_single_cluster) {
    std::vector<int> selected_clusters;
    std::set<int> processed;
    
    for (int leaf : leaves) {
        // Skip if already processed
        if (processed.find(leaf) != processed.end()) {
            continue;
        }
        
        // Find the birth epsilon (1/value) for this leaf
        double eps = 0.0;
        bool found = false;
        
        for (const auto& node : cluster_tree) {
            if (node.child == leaf) {
                eps = 1.0 / node.lambda_val;
                found = true;
                break;
            }
        }
        
        // If we couldn't find the leaf in the tree, skip it
        if (!found) {
            continue;
        }
        
        // Check if epsilon is below threshold
        if (eps < cluster_selection_epsilon) {
            // Need to traverse upwards to find a suitable parent
            int epsilon_child = traverse_upwards(
                cluster_tree,
                cluster_selection_epsilon,
                leaf,
                allow_single_cluster
            );
            
            selected_clusters.push_back(epsilon_child);
            
            // Mark all descendant nodes as processed
            std::vector<int> descendants = bfs_from_cluster_tree(cluster_tree, epsilon_child);
            for (int sub_node : descendants) {
                if (sub_node != epsilon_child) {
                    processed.insert(sub_node);
                }
            }
        } else {
            // Epsilon is sufficient, keep this leaf
            selected_clusters.push_back(leaf);
        }
    }
    
    // Convert vector to set and return
    return std::set<int>(selected_clusters.begin(), selected_clusters.end());
}

// Leaf selection function for the pipeline
std::set<int> leaf_selection(const std::vector<CondensedNode>& condensed_tree,
                           double cluster_selection_epsilon,
                           bool allow_single_cluster) {
    
    // Step 1: Get leaf nodes from cluster tree
    // Leaf Nodes are nodes that aren't parents, so formed from merge of two invalid clusters
    // Can be >= min_cluster_size
    std::set<int> leaves = get_cluster_tree_leaves(condensed_tree);
    
    std::cout << "Found " << leaves.size() << " leaf clusters: ";
    for (int leaf : leaves) {
        std::cout << leaf << " ";
    }
    std::cout << std::endl;
    
    // Step 2: Handle case when no leaves found
    std::set<int> selected_clusters;
    if (leaves.empty()) {
        std::cout << "No leaves found, selecting root cluster" << std::endl;
        int root_cluster = find_min_parent(condensed_tree);
        selected_clusters.insert(root_cluster);
    } else {
        // Step 3: Apply epsilon search if specified
        // Epsilon Search reduces fragmentation
        // By setting a non-zero cluster_selection_epsilon, 
        // clusters which are spawned at an epsilon < cluster_selection_epsilon are merged
        // Until a parent cluster is found with epsilon >= cluster_selection_epsilon
        if (cluster_selection_epsilon != 0.0) {
            std::cout << "Applying epsilon search with epsilon = " << cluster_selection_epsilon << std::endl;
            selected_clusters = epsilon_search(leaves, condensed_tree, cluster_selection_epsilon, allow_single_cluster);
        } else {
            selected_clusters = leaves;
        }
    }
    
    std::cout << "Selected clusters: ";
    for (int cluster : selected_clusters) {
        std::cout << cluster << " ";
    }
    std::cout << std::endl;
    
    return selected_clusters;
}

// Assign labels to points based on selected clusters and return cluster membership
std::vector<std::vector<int>> do_labelling_with_clusters(
    const std::vector<CondensedNode>& condensed_tree,
    const std::set<int>& selected_clusters,
    int n_samples) {

    
    std::vector<int> assignment(n_samples, -1);  // -1 means noise
    std::vector<std::vector<int>> clusters;
    
    std::cout << "[DEBUG] Starting labelling with " << selected_clusters.size() << " selected clusters\n";
    
    // For each selected cluster, find all points that belong to it
    for (int cluster_id : selected_clusters) {
        std::vector<int> this_cluster;
        std::queue<int> to_process;
        to_process.push(cluster_id);
        
        std::cout << "[DEBUG] Processing selected cluster " << cluster_id << "\n";
        
        while (!to_process.empty()) {
            int current_cluster = to_process.front();
            to_process.pop();
            
            // Find all edges from this cluster
            for (const auto& node : condensed_tree) {
                if (node.parent == current_cluster) {
                    if (node.cluster_size == 1) {
                        // This is a point
                        if (node.child < n_samples && assignment[node.child] == -1) {
                            assignment[node.child] = clusters.size();
                            this_cluster.push_back(node.child);
                        }
                    } else {
                        // This is a sub-cluster - only process if it's not selected
                        if (selected_clusters.find(node.child) == selected_clusters.end()) {
                            to_process.push(node.child);
                        }
                    }
                }
            }
        }
        
        if (!this_cluster.empty()) {
            clusters.push_back(std::move(this_cluster));
            std::cout << "[DEBUG] Cluster " << (clusters.size()-1)
                      << " got " << clusters.back().size() << " points\n";
        }
    }
    
    // Count noise points
    int noise_count = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (assignment[i] == -1) {
            noise_count++;
        }
    }
    
    std::cout << "[DEBUG] Found " << clusters.size() << " clusters, " 
              << noise_count << " noise points\n";
    
    return clusters;
}

// Helper class for timing sections
class SectionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string section_name;
    
public:
    SectionTimer(const std::string& name) : section_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[TIMER] Starting: " << section_name << std::endl;
    }
    
    ~SectionTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TIMER] Completed: " << section_name 
                  << " - " << duration.count() << " ms" << std::endl;
    }
};

// Macro for easy timing
#define TIME_SECTION(name) SectionTimer timer(name)

std::vector<std::vector<int>> single_linkage_clustering(
    const std::vector<Edge>& mst_edges,
    int N_pts,
    int min_cluster_size,
    clusterMethod clusterMethod
)
{
    auto overall_start = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== Running Single Linkage Clustering ===" << std::endl;
    std::cout << "[PROFILE] Input size: " << N_pts << " points, " << mst_edges.size() << " edges" << std::endl;

    // ====== INITIALIZATION AND EDGE PROCESSING ======
        
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

    // ====== UNION-FIND INITIALIZATION ======
    TIME_SECTION("Union-Find Initialization");
    
    int max_clusters = 2 * N_pts;
    std::vector<int> parent(max_clusters), sz(max_clusters);
    std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters);
    std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // initialise all points as singleton clusters //
    for(int i = 0; i < N_pts; ++i){
        parent[i] = i;
        sz[i] = 1;
        birth_lambda[i] = lambda_max;
        death_lambda[i] = 0;
    }
    int next_cluster_id = N_pts;

    std::cout << "[DEBUG] Initialized " << next_cluster_id << " singleton clusters\n"; 

    // ====== HIERARCHY BUILDING ======
    {
        TIME_SECTION("Hierarchy Building (Union-Find Operations)");
        
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

        std::cout << "\n=== HIERARCHY BUILDING ===" << std::endl;
        int merge_count = 0;
        for(int edge_idx = 0; edge_idx < edges_copy.size(); ++edge_idx) {
            auto &e = edges_copy[edge_idx];
            
            // if nodes are already in the same cluster, continue
            int c1 = find_root(e.u), c2 = find_root(e.v);
            if(c1 == c2) continue;

            merge_count++;
            // lambda = 1 / MRD - decrease from lambda = infinity to lambda = 0
            float lambda = 1.f / e.weight;
                
            // Set death lambda and record stability
            death_lambda[c1] = lambda;
            death_lambda[c2] = lambda;

            // make new cluster
            int c_new = next_cluster_id++;
            parent[c1] = parent[c2] = c_new;
            parent[c_new] = c_new;
            sz[c_new]           = sz[c1] + sz[c2];
            birth_lambda[c_new] = lambda;
            death_lambda[c_new] = 0;
            left_child[c_new]   = c1;
            right_child[c_new]  = c2;
        }
        
        std::cout << "[PROFILE] Performed " << merge_count << " merges from " << edges_copy.size() << " edges" << std::endl;
        std::cout << "[DEBUG] Total clusters created: " << next_cluster_id << "\n";
    }

    // ====== FIND ROOT CLUSTERS ======
    std::vector<int> root_clusters;
    {
        TIME_SECTION("Finding Root Clusters");
        
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
    }
    
    // ====== CONDENSE TREE FOR MULTIPLE ROOT NODES ======
    std::vector<CondensedNode> combined_condensed_tree;
    {
        TIME_SECTION("Condensed Tree Construction");
        std::cout << "\n=== CONDENSE TREE STEP ===" << std::endl;
        
        // Global label counter to avoid collisions across multiple roots
        int next_label = 0;

        // Process each root cluster separately
        for(int root_node : root_clusters) {
            auto root_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "[DEBUG] Root node for condensing: " << root_node << "\n";
            std::vector<CondensedNode> root_condensed_tree = condense_tree(
                left_child, right_child, birth_lambda, death_lambda, sz, 
                N_pts, root_node, next_label, min_cluster_size
            );
            
            auto root_end = std::chrono::high_resolution_clock::now();
            auto root_duration = std::chrono::duration_cast<std::chrono::milliseconds>(root_end - root_start);
            std::cout << "[PROFILE] Root " << root_node << " condensing took: " << root_duration.count() << " ms" << std::endl;
            
            // Append this root's condensed tree to the combined tree
            combined_condensed_tree.insert(
                combined_condensed_tree.end(), 
                root_condensed_tree.begin(), 
                root_condensed_tree.end()
            );
        }
        
        std::cout << "[PROFILE] Total condensed tree size: " << combined_condensed_tree.size() << " nodes" << std::endl;
    }

    // ====== STABILITY CALCULATION & CLUSTER EXTRACTION ======
    std::map<int, ClusterStability> cluster_stability;
    {
        TIME_SECTION("Stability Calculation");
        std::cout << "\n=== STABILITY CALCULATION ===" << std::endl;
        cluster_stability = calculate_cluster_stability(combined_condensed_tree);
        std::cout << "[PROFILE] Calculated stability for " << cluster_stability.size() << " clusters" << std::endl;
    }
    
    std::set<int> selected_clusters;
    float cluster_selection_epsilon = 0.0f;
    
    {
        TIME_SECTION("Cluster Selection");
        std::cout << "=== CLUSTER EXTRACTION STEP ===" << std::endl;
        
        switch(clusterMethod){
            case clusterMethod::EOM:
                std::cout << "=== CLUSTER METHOD SELECTED: EXCESS OF MASS ===" << std::endl;
                selected_clusters = excess_of_mass_selection(cluster_stability);
                break;
            case clusterMethod::Leaf:
                {
                    std::cout << "=== CLUSTER METHOD SELECTED: LEAF ===" << std::endl;
                    std::cout << "\n=== CALCULATING OPTIMAL CLUSTER SELECTION EPSILON ===" << std::endl;
                    
                    auto epsilon_start = std::chrono::high_resolution_clock::now();
                    cluster_selection_epsilon = 1.0 / calculate_cluster_selection_epsilon(combined_condensed_tree, cluster_stability);
                    auto epsilon_end = std::chrono::high_resolution_clock::now();
                    auto epsilon_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epsilon_end - epsilon_start);
                    std::cout << "[PROFILE] Epsilon calculation took: " << epsilon_duration.count() << " ms" << std::endl;
                    
                    selected_clusters = leaf_selection(combined_condensed_tree, cluster_selection_epsilon);
                }
                break;
            default:
                std::cout << "=== NO VALID CLUSTER METHOD SELECTED ===" << std::endl;
                std::vector<std::vector<int>> fail;
                return fail;
        }
        
        std::cout << "[PROFILE] Selected " << selected_clusters.size() << " clusters" << std::endl;
    }
    
    std::vector<std::vector<int>> final_clusters;
    {
        TIME_SECTION("Final Labelling");
        std::cout << "=== LABELLING CLUSTERS STEP ===" << std::endl;
        final_clusters = do_labelling_with_clusters(combined_condensed_tree, selected_clusters, N_pts);
    }

    // ====== FINAL RESULTS AND TIMING ======
    {
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start);
        
        std::cout << "\n=== FINAL CLUSTERING RESULT ===" << std::endl;
        std::cout << "[PROFILE] TOTAL EXECUTION TIME: " << total_duration.count() << " ms" << std::endl;
        std::cout << "Found " << final_clusters.size() << " clusters:\n";
        for (int i = 0; i < final_clusters.size(); ++i) {
            std::cout << "Cluster " << i << ": " << final_clusters[i].size() << " points\n";
            // Show first few points in each cluster
            std::cout << "  Points: ";
            for (int j = 0; j < std::min(10, (int)final_clusters[i].size()); ++j) {
                std::cout << final_clusters[i][j] << " ";
            }
            if (final_clusters[i].size() > 10) {
                std::cout << "... (+" << (final_clusters[i].size() - 10) << " more)";
            }
            std::cout << "\n";
        }
    }

    return final_clusters;
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