#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <limits>
#include <string>

// Forward declarations
struct Point;
struct Edge;
struct TreeNode;

// Basic data structures
struct Point {
    float* coordinates;  // d-dimensional point
    int id;
    int component_id;
    float core_distance;
    
    __host__ __device__ Point() : coordinates(nullptr), id(-1), component_id(-1), core_distance(std::numeric_limits<float>::max()) {}
};

struct Edge {
    int source;
    int target;
    float weight;
    
    Edge() : source(-1), target(-1), weight(std::numeric_limits<float>::max()) {}
    Edge(int s, int t, float w) : source(s), target(t), weight(w) {}
};

struct TreeNode {
    int* point_indices;      // Points in this node
    int num_points;
    float* bounds_min;       // Bounding box minimum
    float* bounds_max;       // Bounding box maximum
    int left_child;
    int right_child;
    bool is_leaf;
    int component_id;        // -1 if mixed components
    float radius;           // Node radius for pruning
    
    TreeNode() : point_indices(nullptr), num_points(0), bounds_min(nullptr), 
                 bounds_max(nullptr), left_child(-1), right_child(-1), 
                 is_leaf(false), component_id(-1), radius(0.0f) {}
};

class UnionFind {
private:
    int* h_parent;
    int* h_rank;
    bool* h_is_component;
    int* d_parent;
    int* d_rank;
    bool* d_is_component;
    int size;

public:
    UnionFind(int n) : size(n) {
        // Allocate host memory
        h_parent = new int[size];
        h_rank = new int[size];
        h_is_component = new bool[size];
        
        // Initialize host arrays
        for (int i = 0; i < size; i++) {
            h_parent[i] = i;           // Each element is its own parent initially
            h_rank[i] = 0;             // All ranks start at 0
            h_is_component[i] = true;  // All elements are component roots initially
        }
        
        // Allocate device memory
        hipMalloc(&d_parent, size * sizeof(int));
        hipMalloc(&d_rank, size * sizeof(int));
        hipMalloc(&d_is_component, size * sizeof(bool));
        
        // Copy initial data to device
        copy_to_device();
    }
    
    ~UnionFind() {
        delete[] h_parent;
        delete[] h_rank;
        delete[] h_is_component;
        hipFree(d_parent);
        hipFree(d_rank);
        hipFree(d_is_component);
    }
    
    // Find with path compression (iterative version matching original)
    int find(int x) {
        int x_parent = h_parent[x];
        int x_grandparent;
        
        while (x_parent != x) {
            x_grandparent = h_parent[x_parent];
            h_parent[x] = x_grandparent;  // Path compression
            x = x_parent;
            x_parent = x_grandparent;
        }
        return x;
    }
    
    // Union by rank (matching original logic)
    void union_sets(int x, int y) {
        int x_root = find(x);
        int y_root = find(y);
        
        if (x_root == y_root) return;  // Already in same component
        
        // Union by rank
        if (h_rank[x_root] < h_rank[y_root]) {
            h_parent[x_root] = y_root;
            h_is_component[x_root] = false;
        } else if (h_rank[x_root] > h_rank[y_root]) {
            h_parent[y_root] = x_root;
            h_is_component[y_root] = false;
        } else {
            h_rank[x_root]++;  // Increase rank when trees have same height
            h_parent[y_root] = x_root;
            h_is_component[y_root] = false;
        }
    }
    
    bool same_component(int x, int y) {
        return find(x) == find(y);
    }
    
    // Get all component roots
    std::vector<int> components() {
        std::vector<int> result;
        for (int i = 0; i < size; i++) {
            if (h_is_component[i]) {
                result.push_back(i);
            }
        }
        return result;
    }
    
    // GPU memory management
    void copy_to_device() {
        hipMemcpy(d_parent, h_parent, size * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_rank, h_rank, size * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_is_component, h_is_component, size * sizeof(bool), hipMemcpyHostToDevice);
    }
    
    void copy_from_device() {
        hipMemcpy(h_parent, d_parent, size * sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(h_rank, d_rank, size * sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(h_is_component, d_is_component, size * sizeof(bool), hipMemcpyDeviceToHost);
    }
    
    // Getters for device pointers (for GPU kernels)
    int* get_d_parent() { return d_parent; }
    int* get_d_rank() { return d_rank; }
    bool* get_d_is_component() { return d_is_component; }
};

// **MAIN CLASS - Drop-in replacement for KDTreeBoruvkaAlgorithm**
class HIPSingleTreeBoruvka {
private:
    // Core data matching original implementation
    std::vector<std::vector<float>> raw_data;
    std::vector<Point> h_points;
    std::vector<TreeNode> h_tree_nodes;
    std::vector<Edge> h_edges;
    std::unique_ptr<UnionFind> union_find;
    
    // Device data pointers
    Point* d_points;
    TreeNode* d_tree_nodes;
    Edge* d_candidate_edges;
    float* d_distances;
    int* d_component_counts;
    
    // Algorithm parameters matching original
    int num_points;
    int num_features;  // matches num_features in original
    int min_samples;
    float alpha;
    int leaf_size;
    bool approx_min_span_tree;
    int n_jobs;  // For CPU fallback compatibility
    std::string metric;
    
    // Current algorithm state
    std::vector<int> components;
    std::vector<int> candidate_neighbor;
    std::vector<int> candidate_point;
    std::vector<float> candidate_distance;
    std::vector<float> core_distance;
    std::vector<float> bounds;
    
    // HIP execution configuration
    dim3 block_size;
    dim3 grid_size;
    
public:
    // **EXACT SAME CONSTRUCTOR SIGNATURE AS ORIGINAL**
    HIPSingleTreeBoruvka(const std::vector<std::vector<float>>& tree_data,
                         int min_samples = 5,
                         const std::string& metric = "euclidean", 
                         int leaf_size = 20,
                         float alpha = 1.0,
                         bool approx_min_span_tree = false,
                         int n_jobs = 4);
    
    ~HIPSingleTreeBoruvka();
    
    // **EXACT SAME PUBLIC METHOD AS ORIGINAL**
    // Returns numpy-style array: [[source, target, weight], ...]
    std::vector<std::vector<float>> spanning_tree();
    
    // **MATCHING PUBLIC ATTRIBUTES FOR COMPATIBILITY**
    std::vector<float> get_core_distance() const { return core_distance; }
    std::vector<float> get_bounds() const { return bounds; }
    std::vector<int> get_candidate_neighbor() const { return candidate_neighbor; }
    std::vector<int> get_candidate_point() const { return candidate_point; }
    std::vector<float> get_candidate_distance() const { return candidate_distance; }
    int get_num_edges() const { return h_edges.size(); }
    
private:
    // Core algorithm phases (internal methods)
    void initialize_components();
    void build_tree();
    void compute_bounds();  // matches _compute_bounds() in original
    int update_components(); // matches update_components() in original
    void compute_core_distances();
    bool find_minimum_edges();
    
    // Tree traversal (replaces dual_tree_traversal with single tree)
    void single_tree_traversal(int node_idx);
    
    // HIP memory management
    void allocate_device_memory();
    void free_device_memory();
    void copy_to_device();
    void copy_from_device();
    
    // Tree construction helpers
    void recursive_build_tree(int node_idx, std::vector<int>& point_indices);
    int find_split_dimension(const std::vector<int>& point_indices);
    void compute_node_bounds(int node_idx);
    void update_tree_components();
    
    // Distance computation helpers
    void launch_core_distance_kernel();
    void launch_edge_finding_kernel();
    
    // Utility functions matching original behavior
    float compute_mutual_reachability_distance(int p1, int p2) const;
    bool can_prune_node(int node_idx, int target_component) const;
    float compute_distance(int p1, int p2) const;
    float rdist_to_dist(float rdist) const;
    float dist_to_rdist(float dist) const;
};

// HIP Kernels
__global__ void compute_core_distances_kernel(
    Point* points, 
    TreeNode* tree_nodes,
    int num_points,
    int min_samples,
    int num_features,
    float* knn_distances
);

__global__ void find_minimum_edges_kernel(
    Point* points,
    TreeNode* tree_nodes, 
    Edge* candidate_edges,
    int* component_list,
    int num_components,
    int num_features,
    float alpha
);

__global__ void update_component_labels_kernel(
    Point* points,
    int* parent_array,
    int num_points
);

__global__ void initialize_candidates_kernel(
    Point* points,
    int* candidate_neighbor,
    int* candidate_point,
    float* candidate_distance,
    int num_points
);

// Device utility functions
__device__ float compute_euclidean_distance(
    const Point& p1, 
    const Point& p2, 
    int num_features
);

__device__ float compute_mutual_reachability_dist(
    const Point& p1, 
    const Point& p2, 
    int num_features,
    float alpha
);

__device__ void traverse_tree_for_component(
    int query_component,
    TreeNode* tree_nodes,
    Point* points,
    int root_node,
    int* best_neighbor,
    int* best_point,
    float* best_distance
);

// **PYTHON BINDING INTERFACE** (if you want to use it from Python)
extern "C" {
    // C interface that matches the original Cython interface
    void* create_hip_boruvka(float* data, int n_points, int n_features,
                            int min_samples, float alpha, int leaf_size,
                            int approx_min_span_tree, int n_jobs);
    
    void destroy_hip_boruvka(void* instance);
    
    float* compute_spanning_tree(void* instance, int* num_edges);
    
    float* get_core_distances(void* instance);
}

// Implementation structure and learning guide:

/*
COMPATIBILITY CHECKLIST:

✓ Constructor matches: HIPSingleTreeBoruvka(tree_data, min_samples, metric, leaf_size, alpha, approx_min_span_tree, n_jobs)
✓ Main method matches: spanning_tree() returns edges array
✓ Attributes match: core_distance, bounds, candidate_*, num_edges
✓ Internal state matches: components, union_find structure
✓ Algorithm flow matches: initialize -> compute_bounds -> loop(find_edges -> update_components)

KEY IMPLEMENTATION POINTS:

1. DATA COMPATIBILITY:
   - Input: std::vector<std::vector<float>> (same as sklearn tree.data)
   - Output: std::vector<std::vector<float>> shaped like [[source, target, weight], ...]
   - Internal arrays match original sizes and meanings

2. ALGORITHM FLOW COMPATIBILITY:
   spanning_tree() {
       initialize_components();
       compute_bounds();  // Includes core distance computation
       
       while (components.size() > 1) {
           single_tree_traversal(0);  // Replaces dual_tree_traversal(0, 0)
           num_components = update_components();
           if (num_components == previous_num && approx_min_span_tree) break;
       }
       
       return edges_as_array();
   }

3. METRIC COMPATIBILITY:
   - Support 'euclidean' metric initially
   - Use rdist (squared distance) internally for performance
   - Convert back to dist for final output

4. MEMORY LAYOUT COMPATIBILITY:
   - Points stored as contiguous float arrays
   - Tree nodes follow sklearn's NodeData_t structure
   - Component tracking matches BoruvkaUnionFind behavior

5. PERFORMANCE OPTIMIZATIONS:
   - GPU parallel processing of multiple components
   - Shared memory for tree traversal
   - Coalesced memory access patterns
   - Early termination with approx_min_span_tree

TESTING STRATEGY:
1. Unit test with identical input/output to original
2. Verify core distances match exactly
3. Check edge weights are identical
4. Validate component merging sequence
5. Performance benchmark against original

INTEGRATION POINTS:
- Can be used as drop-in replacement in HDBSCAN
- Maintains same error handling behavior
- Supports same debugging/profiling interfaces
- Compatible with existing Python bindings
*/

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Example usage matching original:
/*
// Python-like usage:
auto data = load_your_data();  // std::vector<std::vector<float>>
HIPSingleTreeBoruvka mst_algo(data, 5, "euclidean", 20, 1.0, false, 4);
auto edges = mst_algo.spanning_tree();  // Returns [[source, target, weight], ...]

// Access attributes like original:
auto core_dists = mst_algo.get_core_distance();
auto bounds = mst_algo.get_bounds();
*/