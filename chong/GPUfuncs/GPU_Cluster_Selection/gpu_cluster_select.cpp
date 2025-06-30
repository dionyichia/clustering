#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

// Simplified data structure for GPU
struct GPUClusterChoice {
    float total_stability;
    int selection_type;  // 0 = invalid, 1 = select_self, 2 = select_descendants
    
    __host__ __device__
    GPUClusterChoice(float stab = 0.0f, int type = 0) 
        : total_stability(stab), selection_type(type) {}
};

// GPU kernel to process one level of the tree
__global__ void process_level_kernel(
    int* nodes_in_level,     // Array of node IDs in this level
    int num_nodes_in_level,  // Size of current level
    float* stability,        // stability[c] for each cluster c
    int* sz,                 // sz[c] for each cluster c  
    int* left_child,         // left_child[c] for each cluster c
    int* right_child,        // right_child[c] for each cluster c
    GPUClusterChoice* dp,    // DP table (input: children, output: current level)
    int min_cluster_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_in_level) return;
    
    int c = nodes_in_level[idx];
    
    // Skip clusters smaller than minimum size
    if (sz[c] < min_cluster_size) {
        dp[c] = GPUClusterChoice(0.0f, 0);  // Invalid
        return;
    }
    
    int L = left_child[c];
    int R = right_child[c];
    
    if (L == -1 && R == -1) {
        // Leaf node: only choice is to select self
        dp[c] = GPUClusterChoice(stability[c], 1);  // select_self
    } else {
        // Internal node: compare self vs descendants
        float self_stability = stability[c];
        float descendants_stability = 0.0f;
        
        // Aggregate from valid children
        if (L >= 0 && sz[L] >= min_cluster_size && dp[L].selection_type != 0) {
            descendants_stability += dp[L].total_stability;
        }
        if (R >= 0 && sz[R] >= min_cluster_size && dp[R].selection_type != 0) {
            descendants_stability += dp[R].total_stability;
        }
        
        // Choose the better option
        if (self_stability >= descendants_stability) {
            dp[c] = GPUClusterChoice(self_stability, 1);  // select_self
        } else {
            dp[c] = GPUClusterChoice(descendants_stability, 2);  // select_descendants
        }
    }
}

// Host function to organize and execute GPU computation
std::vector<int> gpu_cluster_selection(
    const std::vector<float>& stability,
    const std::vector<int>& sz,
    const std::vector<int>& left_child,
    const std::vector<int>& right_child,
    int N_pts,
    int next_cluster_id,
    int min_cluster_size
) {
    // 1. Build level structure on CPU
    std::vector<std::vector<int>> levels;
    std::vector<int> level_of_node(next_cluster_id, -1);
    
    // Assign levels (BFS from leaves)
    std::queue<int> q;
    for (int c = 0; c < N_pts; ++c) {  // Leaf nodes
        level_of_node[c] = 0;
        if (levels.size() <= 0) levels.resize(1);
        levels[0].push_back(c);
        q.push(c);
    }
    
    // Build level structure
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        for (int p = N_pts; p < next_cluster_id; ++p) {
            if ((left_child[p] == current || right_child[p] == current) && 
                level_of_node[p] == -1) {
                
                int left_level = (left_child[p] >= 0) ? level_of_node[left_child[p]] : -1;
                int right_level = (right_child[p] >= 0) ? level_of_node[right_child[p]] : -1;
                
                if (left_level != -1 && right_level != -1) {
                    int parent_level = std::max(left_level, right_level) + 1;
                    level_of_node[p] = parent_level;
                    
                    if (levels.size() <= parent_level) levels.resize(parent_level + 1);
                    levels[parent_level].push_back(p);
                    q.push(p);
                }
            }
        }
    }
    
    // 2. Copy data to GPU
    thrust::device_vector<float> d_stability(stability);
    thrust::device_vector<int> d_sz(sz);
    thrust::device_vector<int> d_left_child(left_child);
    thrust::device_vector<int> d_right_child(right_child);
    thrust::device_vector<GPUClusterChoice> d_dp(next_cluster_id);
    
    // 3. Process each level on GPU
    for (size_t level = 0; level < levels.size(); ++level) {
        if (levels[level].empty()) continue;
        
        // Copy current level's nodes to GPU
        thrust::device_vector<int> d_nodes_in_level(levels[level]);
        
        // Launch kernel for this level
        int num_nodes = levels[level].size();
        int block_size = 256;
        int num_blocks = (num_nodes + block_size - 1) / block_size;
        
        process_level_kernel<<<num_blocks, block_size>>>(
            thrust::raw_pointer_cast(d_nodes_in_level.data()),
            num_nodes,
            thrust::raw_pointer_cast(d_stability.data()),
            thrust::raw_pointer_cast(d_sz.data()),
            thrust::raw_pointer_cast(d_left_child.data()),
            thrust::raw_pointer_cast(d_right_child.data()),
            thrust::raw_pointer_cast(d_dp.data()),
            min_cluster_size
        );
        
        // Synchronize after each level
        cudaDeviceSynchronize();
    }
    
    // 4. Copy results back to CPU
    thrust::host_vector<GPUClusterChoice> h_dp = d_dp;
    
    // 5. Extract final clusters using CPU logic
    std::vector<int> final_clusters;
    
    // Find root and extract selection
    for (int c = 0; c < next_cluster_id; ++c) {
        if (level_of_node[c] == -1) continue;  // Skip invalid nodes
        
        bool is_root = true;
        for (int p = N_pts; p < next_cluster_id; ++p) {
            if (left_child[p] == c || right_child[p] == c) {
                is_root = false;
                break;
            }
        }
        
        if (is_root && sz[c] >= min_cluster_size) {
            // Recursively extract selected clusters
            extract_selected_clusters_recursive(c, h_dp, left_child, right_child, final_clusters);
        }
    }
    
    return final_clusters;
}

// Recursive helper to extract final selection
void extract_selected_clusters_recursive(
    int c,
    const thrust::host_vector<GPUClusterChoice>& dp,
    const std::vector<int>& left_child,
    const std::vector<int>& right_child,
    std::vector<int>& final_clusters
) {
    if (dp[c].selection_type == 1) {  // select_self
        final_clusters.push_back(c);
    } else if (dp[c].selection_type == 2) {  // select_descendants
        int L = left_child[c], R = right_child[c];
        if (L >= 0) extract_selected_clusters_recursive(L, dp, left_child, right_child, final_clusters);
        if (R >= 0) extract_selected_clusters_recursive(R, dp, left_child, right_child, final_clusters);
    }
}