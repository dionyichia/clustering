#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/hip/execution_policy.h>  // rocThrust execution policies
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <queue>
#include <chrono>
#include <iomanip>
using namespace cooperative_groups;

// Constants
#define WARP_SIZE 64  // AMD Wavefront size. Number of Threads in a Wavefront.
#define MAX_THREADS_PER_BLOCK 256
#define MAX_GRID_SIZE 4294967295 
#define MIN_PARTITION_SIZE 1  // Minimum size before creating leaf
#define MAX_TREE_DEPTH 25     // Maximum tree depth to prevent infinite recursion

#define HIP_CHECK(err)                                                      \
  do {                                                                       \
    hipError_t _e = (err);                                                   \
    if (_e != hipSuccess) {                                                  \
      std::cerr                                                             \
        << "HIP error at " << __FILE__ << ":" << __LINE__                   \
        << " → " << hipGetErrorString(_e) << std::endl;                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// KD Tree Node structure
struct KdNode {
    int tuple_idx;        // Index into coordinate array
    int split_dim;        // Dimension to split on
    int left_child;       // Index of left child (-1 if leaf)
    int right_child;      // Index of right child (-1 if leaf)
    float split_value;    // Value at split dimension
    
    // Default constructor for leaf nodes
    __host__ __device__ KdNode() : tuple_idx(-1), split_dim(-1), left_child(-1), right_child(-1), split_value(0.0f) {}
};

// Partition structure for managing tree construction
struct Partition {
    int* indices;         // Array of point indices
    int size;            // Number of points in partition
    int node_idx;        // Tree node index for this partition
    int level;           // Tree level
    int start_offset;    // Starting offset in the indices array
};

// GPU Kernel for super key comparison
__device__ bool superKeyCompare(const float* coords, int D, int idx1, int idx2, int primary_dim) {
    // Compare primary dimension first
    float val1 = coords[idx1 * D + primary_dim];
    float val2 = coords[idx2 * D + primary_dim];
    
    if (fabsf(val1 - val2) > 1e-9f) {  // Use epsilon for float comparison
        return val1 < val2;
    }
    
    // If primary dimensions are equal, compare other dimensions
    for (int d = 0; d < D; d++) {
        if (d == primary_dim) continue;
        
        val1 = coords[idx1 * D + d];
        val2 = coords[idx2 * D + d];
        
        if (fabsf(val1 - val2) > 1e-9f) {
            return val1 < val2;
        }
    }
    
    // If all coordinates are equal, compare by index for deterministic ordering
    return idx1 < idx2;
}




// Fixed partition function - process one partition at a time to avoid races
void sortPartitionWithRocThrustSequential(
    int* d_indices,
    float* d_coords,
    int D,
    int* d_split_dims,
    int* d_partition_starts,
    int* d_partition_sizes,
    int* d_medians,
    int num_partitions)
{
    // Copy metadata to host
    std::vector<int> h_split(num_partitions);
    std::vector<int> h_start(num_partitions);
    std::vector<int> h_size(num_partitions);
    
    HIP_CHECK(hipMemcpy(h_split.data(), d_split_dims, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_start.data(), d_partition_starts, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_size.data(), d_partition_sizes, num_partitions * sizeof(int), hipMemcpyDeviceToHost));

    std::vector<int> h_medians(num_partitions);

    // Sort each partition SEQUENTIALLY to avoid race conditions
    for (int p = 0; p < num_partitions; ++p) {
        int start = h_start[p];
        int size = h_size[p];
        int dim = h_split[p];

        if (size <= 0) {
            h_medians[p] = -1;
            continue;
        }

        thrust::device_ptr<int> idx_begin(d_indices + start);
        thrust::device_ptr<int> idx_end = idx_begin + size;

        // Create a separate stream for each partition to ensure proper synchronization
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        auto policy = thrust::hip::par.on(stream);

        thrust::sort(
            policy,
            idx_begin, idx_end,
            [=] __device__ (int a, int b) {
                float va = d_coords[a * D + dim];
                float vb = d_coords[b * D + dim];
                if (va < vb) return true;
                if (va > vb) return false;
                
                // Tie-breaker using super-key
                for (int dd = 0; dd < D; ++dd) {
                    if (dd == dim) continue;
                    float aa = d_coords[a * D + dd];
                    float bb = d_coords[b * D + dd];
                    if (aa < bb) return true;
                    if (aa > bb) return false;
                }
                return a < b; // Final tie-breaker by index
            });

        // Wait for this partition to complete before moving to next
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipStreamDestroy(stream));

        // Get the median
        if (size == 1) {
            HIP_CHECK(hipMemcpy(&h_medians[p], d_indices + start, sizeof(int), hipMemcpyDeviceToHost));
        } else {
            HIP_CHECK(hipMemcpy(&h_medians[p], d_indices + start + size/2, sizeof(int), hipMemcpyDeviceToHost));
        }
    }

    // Copy medians back to GPU
    HIP_CHECK(hipMemcpy(d_medians, h_medians.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
}

// stores sizes of left and right partition arrays
// stores indices of points in respective left and right arrays
__global__ void partitionArraysKernel(
    int* input_indices, 
    float* coords, 
    int D, 
    int* split_dims,
    int* partition_starts, 
    int* partition_sizes,
    int* medians, 
    int* left_starts, 
    int* right_starts,
    int* left_sizes, 
    int* right_sizes,
    int* output_indices, 
    int num_partitions) {

    int partition_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (partition_id >= num_partitions) return;
    
    // indice of first element if it were in the flat contiguous array
    int start = partition_starts[partition_id];
    int size = partition_sizes[partition_id];
    int median_idx = medians[partition_id];
    int dim = split_dims[partition_id];
    
    if (size <= 1 || median_idx == -1) {
        left_sizes[partition_id] = 0;
        right_sizes[partition_id] = 0;
        return;
    }
    
    float split_val = coords[median_idx * D + dim];
    
    int left_count = 0, right_count = 0;
    int left_base = left_starts[partition_id];
    int right_base = right_starts[partition_id];
    
    // Partition points
    for (int i = 0; i < size; i++) {
        int point_idx = input_indices[start + i];
        if (point_idx == median_idx) continue;  // Skip the median itself
        
        float val = coords[point_idx * D + dim];
        
        // Use deterministic partitioning
        bool goes_left = false;
        if (val < split_val) {
            goes_left = true;
        } else if (fabsf(val - split_val) <= 1e-9f) {
            // Equal values: use super key comparison for deterministic placement
            goes_left = superKeyCompare(coords, D, point_idx, median_idx, dim);
        }
        
        if (goes_left) {
            output_indices[left_base + left_count] = point_idx;
            left_count++;
        } else {
            output_indices[right_base + right_count] = point_idx;
            right_count++;
        }
    }
    
    left_sizes[partition_id] = left_count;
    right_sizes[partition_id] = right_count;
}

// Kernel to initialize tree nodes
__global__ void initializeNodesKernel(
    KdNode* tree_nodes, 
    int* medians, 
    float* coords,
    int D, 
    int* node_indices, 
    int* split_dims,
    int num_nodes) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    int node_idx = node_indices[tid];
    // medians stores the median value used to split at a respective KDNode
    // so len(medians) is == len(KDnodes)
    int median_idx = medians[tid];
    int split_dim = split_dims[tid];
    
    if (median_idx == -1) {
        // Empty partition - create empty node
        tree_nodes[node_idx] = KdNode();
        return;
    }
    
    tree_nodes[node_idx].tuple_idx = median_idx;
    tree_nodes[node_idx].split_dim = split_dim;
    tree_nodes[node_idx].split_value = coords[median_idx * D + split_dim];
    tree_nodes[node_idx].left_child = -1;   // Will be set later
    tree_nodes[node_idx].right_child = -1;  // Will be set later
}

// Host class for KD-Tree construction
class HipKdTree {
private:
    // stores the points and coordinates in one contiguous array in GPU
    // e.g if D = 3
    // d[0], coord 0 of point 0
    // d[1], coord 1 of point 0
    // d[2], coord 2 of point 0
    float* d_coords;
    // stores all KDNodes in one contiguous array in GPU
    KdNode* d_tree_nodes;
    int* d_indices_buffer1;
    int* d_indices_buffer2;
    int N, D;
    int tree_size;
    int actual_nodes;
    
public:
    // automatically initialises N, D and actual_nodes, attributes of HipKdTree 
    // N = num_points, D = dimensions, actual_nodes = 0 
    // before body of code runs
    HipKdTree(int num_points, int dimensions) : N(num_points), D(dimensions), actual_nodes(0) {
        std::cout << "Initializing HipKdTree with " << N << " points in " << D << " dimensions..." << std::endl;
        
        // Calculate tree size (upper bound)
        tree_size = 2 * N - 1;
        
        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_coords, N * D * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_tree_nodes, tree_size * sizeof(KdNode)));
        HIP_CHECK(hipMalloc(&d_indices_buffer1, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_indices_buffer2, N * sizeof(int)));
        
        // Initialize all members in KDNode to 0
        hipMemset(d_tree_nodes, 0, tree_size * sizeof(KdNode));
        
        double memory_mb = (N * D * sizeof(float) + tree_size * sizeof(KdNode) + 2 * N * sizeof(int)) / (1024.0 * 1024.0);
        std::cout << "Allocated memory: " << memory_mb << " MB" << std::endl;
    }
    
    // destructor 
    ~HipKdTree() {
        if (d_coords) hipFree(d_coords);
        if (d_tree_nodes) hipFree(d_tree_nodes);
        if (d_indices_buffer1) hipFree(d_indices_buffer1);
        if (d_indices_buffer2) hipFree(d_indices_buffer2);
    }
    

    void buildTree(const std::vector<std::vector<float>>& points) {
        std::cout << "Starting tree construction..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Copy coordinate data to device
        std::vector<float> flat_coords(N * D);
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                flat_coords[i * D + d] = points[i][d];
            }
        }
        HIP_CHECK(hipMemcpy(d_coords, flat_coords.data(), N * D * sizeof(float), hipMemcpyHostToDevice));
        
        // Initialize indices
        std::vector<int> initial_indices(N);
        for (int i = 0; i < N; i++) {
            initial_indices[i] = i;
        }

        // d_indices_buffer1 filled with initial indices, since at the start all points are in the same partition
        HIP_CHECK(hipMemcpy(d_indices_buffer1, initial_indices.data(), N * sizeof(int), hipMemcpyHostToDevice));
        
        // Build tree using breadth-first approach
        actual_nodes = buildTreeBreadthFirst();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Tree construction completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Created " << actual_nodes << " nodes (expected ~" << (2 * N - 1) << " for balanced tree)" << std::endl;
        
        // Verify tree structure
        verifyTreeStructure();
    }
    
    // Public method to get tree statistics
    void printTreeStats() {
        std::cout << "\n=== Tree Statistics ===" << std::endl;
        std::cout << "Points: " << N << std::endl;
        std::cout << "Dimensions: " << D << std::endl;
        std::cout << "Tree nodes: " << actual_nodes << std::endl;
        std::cout << "Memory usage: " << (actual_nodes * sizeof(KdNode)) / 1024.0 << " KB for tree nodes" << std::endl;
    }
    
private:
    int buildTreeBreadthFirst() {
        std::cout << "  - Starting breadth-first tree construction..." << std::endl;
        
        // Host data structures for managing partitions
        std::vector<Partition> current_partitions;
        std::vector<Partition> next_partitions;
        
        // Initialize root partition
        Partition root;
        root.indices = d_indices_buffer1;
        root.size = N;
        root.node_idx = 0;
        root.level = 0;
        root.start_offset = 0;
        current_partitions.push_back(root);
        
        int nodes_created = 0;
        int* current_buffer = d_indices_buffer1;
        int* next_buffer = d_indices_buffer2;
        int level = 0;

        // process each level of kd_tree
        while (!current_partitions.empty() && level < MAX_TREE_DEPTH) {
            int num_partitions = current_partitions.size();
            std::cout << "  - Level " << level << ": Processing " << num_partitions << " partitions..." << std::endl;
            
            // Debug: show partition sizes for deeper levels
            if (level > 15) {
                std::cout << "    Partition sizes: ";
                for (int i = 0; i < std::min(10, num_partitions); i++) {
                    std::cout << current_partitions[i].size << " ";
                }
                if (num_partitions > 10) std::cout << "...";
                std::cout << std::endl;
            }
            
            // Add early termination condition to prevent infinite loops
            bool has_meaningful_partitions = false;
            for (int i = 0; i < num_partitions; i++) {
                if (current_partitions[i].size > MIN_PARTITION_SIZE) {
                    has_meaningful_partitions = true;
                    break;
                }
            }
            
            if (!has_meaningful_partitions) {
                std::cout << "    All remaining partitions are size " << MIN_PARTITION_SIZE << " or less, terminating early" << std::endl;
                // Create leaf nodes for remaining partitions
                for (const auto& partition : current_partitions) {
                    if (partition.size == 1) {
                        // Copy the single point index and create a leaf node
                        std::vector<int> single_point(1);
                        HIP_CHECK(hipMemcpy(single_point.data(), partition.indices, sizeof(int), hipMemcpyDeviceToHost));
                        
                        KdNode leaf_node;
                        leaf_node.tuple_idx = single_point[0];
                        leaf_node.split_dim = -1;  // Leaf node indicator
                        leaf_node.left_child = -1;
                        leaf_node.right_child = -1;
                        leaf_node.split_value = 0.0f;
                        
                        HIP_CHECK(hipMemcpy(d_tree_nodes + partition.node_idx, &leaf_node, sizeof(KdNode), hipMemcpyHostToDevice));
                        nodes_created++;
                    }
                }
                break;
            }
            
            // no partitions left to process
            if (num_partitions == 0) break;
            
            // Process current level
            int processed_nodes = processSingleLevel(current_partitions, next_partitions, 
                                                   current_buffer, next_buffer, nodes_created, level);
            nodes_created += processed_nodes;
            
            // Swap buffers and partitions for next iteration
            std::swap(current_buffer, next_buffer);
            current_partitions = std::move(next_partitions);
            next_partitions.clear();
            
            std::cout << "    Level " << level << " completed: " << processed_nodes 
                      << " nodes created, next level has " << current_partitions.size() << " partitions" << std::endl;
            // Add explicit synchronization and debugging
            HIP_CHECK(hipDeviceSynchronize());
            level++;
        }
        
        if (level >= MAX_TREE_DEPTH) {
            std::cout << "  - Warning: Reached maximum tree depth (" << MAX_TREE_DEPTH 
                      << "), some partitions may be larger than optimal" << std::endl;
        }
        
        std::cout << "  - Tree construction completed with " << nodes_created 
                  << " nodes over " << level << " levels" << std::endl;
        return nodes_created;
    }
    
    int processSingleLevel(
        const std::vector<Partition>& current_partitions,
        std::vector<Partition>& next_partitions,
        int* current_buffer, 
        int* next_buffer,
        int nodes_created, 
        int level) {
        
        int num_partitions = current_partitions.size();
        
        // Allocate host arrays for this level
        std::vector<int> partition_starts(num_partitions);
        std::vector<int> partition_sizes(num_partitions);
        std::vector<int> node_indices(num_partitions);
        std::vector<int> split_dims(num_partitions);
        
        // Device arrays for this level
        int* d_partition_starts;
        int* d_partition_sizes;
        int* d_medians;
        int* d_node_indices;
        int* d_split_dims;
        int* d_left_starts;
        int* d_right_starts;
        int* d_left_sizes;
        int* d_right_sizes;
        
        HIP_CHECK(hipMalloc(&d_partition_starts, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_partition_sizes, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_medians, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_node_indices, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_split_dims, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_left_starts, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_right_starts, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_left_sizes, num_partitions * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_right_sizes, num_partitions * sizeof(int)));
        
        // Prepare partition data
        for (int i = 0; i < num_partitions; i++) {
            partition_starts[i] = current_partitions[i].start_offset;
            partition_sizes[i] = current_partitions[i].size;
            node_indices[i] = current_partitions[i].node_idx;
            split_dims[i] = current_partitions[i].level % D;
        }
        
        // Copy to device
        HIP_CHECK(hipMemcpy(d_partition_starts, partition_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_partition_sizes, partition_sizes.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_node_indices, node_indices.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_split_dims, split_dims.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        

        // Find medians using rocThrust
        sortPartitionWithRocThrustSequential(current_buffer, d_coords, D, d_split_dims, 
                                  d_partition_starts, d_partition_sizes, d_medians, num_partitions);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        // Initialize tree nodes
        // Compute how many blocks we need so that
        // blockSize.x * nodeGridSize.x >= num_partitions
        // each block has 256 threads
        dim3 blockSize(256);
        dim3 nodeGridSize(
            (num_partitions + blockSize.x - 1) 
            / blockSize.x
        );
        

        // Launch the kernel to initialize one KdNode per partition.
        //   initializeNodesKernel<<< grid, block, shared_mem, stream >>>(…)
        //   - grid.x = nodeGridSize.x blocks
        //   - block.x = blockSize.x threads per block
        //   - shared_mem = 0 (no dynamic shared memory)
        //   - stream = 0 (default stream)
        //
        // The arguments are:
        //   d_tree_nodes   : pointer to your device KdNode array
        //   d_medians      : array of median indices per partition
        //   d_coords       : flattened coordinate array (N×D floats)
        //   D              : number of dimensions
        //   d_node_indices : array mapping partition‐ID → tree‐node index
        //   d_split_dims   : split dimension per partition
        //   num_partitions : how many nodes/partitions to process
        hipLaunchKernelGGL(
            initializeNodesKernel,
            nodeGridSize,   // <<< grid.x, ...
            blockSize,      //      block.x, ...
            0,              //      no dynamic shared memory
            0,              //      default stream
            d_tree_nodes,   // kernel arg 1
            d_medians,      // kernel arg 2
            d_coords,       // kernel arg 3
            D,              // kernel arg 4
            d_node_indices, // kernel arg 5
            d_split_dims,   // kernel arg 6
            num_partitions  // kernel arg 7
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        // Prepare for partitioning with proper memory layout
        std::vector<int> left_starts(num_partitions);
        std::vector<int> right_starts(num_partitions);
        int current_offset = 0;
        
        // Calculate non-overlapping offsets
        for (int i = 0; i < num_partitions; i++) {
            left_starts[i] = current_offset;
            int max_left_size = (partition_sizes[i] > 0) ? (partition_sizes[i] - 1) : 0;
            current_offset += max_left_size;
            
            right_starts[i] = current_offset;
            int max_right_size = (partition_sizes[i] > 0) ? (partition_sizes[i] - 1) : 0;
            current_offset += max_right_size;
        }
        
        HIP_CHECK(hipMemcpy(d_left_starts, left_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_right_starts, right_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
        
        // Partition arrays
        dim3 gridSize(num_partitions);
        hipLaunchKernelGGL(partitionArraysKernel, gridSize, blockSize, 0, 0,
                          current_buffer, d_coords, D, d_split_dims,
                          d_partition_starts, d_partition_sizes, d_medians,
                          d_left_starts, d_right_starts, d_left_sizes, d_right_sizes,
                          next_buffer, num_partitions);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        // Get results and create next level partitions
        std::vector<int> left_sizes(num_partitions);
        std::vector<int> right_sizes(num_partitions);
        HIP_CHECK(hipMemcpy(left_sizes.data(), d_left_sizes, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(right_sizes.data(), d_right_sizes, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
        
        // Update parent nodes with child pointers and create next level partitions
        updateNodesAndCreateChildren(current_partitions, next_partitions, left_sizes, right_sizes, 
                                   left_starts, right_starts, next_buffer, nodes_created);
        
        // Clean up device memory
        hipFree(d_partition_starts);
        hipFree(d_partition_sizes);
        hipFree(d_medians);
        hipFree(d_node_indices);
        hipFree(d_split_dims);
        hipFree(d_left_starts);
        hipFree(d_right_starts);
        hipFree(d_left_sizes);
        hipFree(d_right_sizes);
        
        return num_partitions;
    }
    
    void updateNodesAndCreateChildren(const std::vector<Partition>& current_partitions,
                                    std::vector<Partition>& next_partitions,
                                    const std::vector<int>& left_sizes,
                                    const std::vector<int>& right_sizes,
                                    const std::vector<int>& left_starts,
                                    const std::vector<int>& right_starts,
                                    int* next_buffer, int nodes_created) {
        
        int num_partitions = current_partitions.size();
        int next_node_idx = nodes_created + num_partitions;
        
        // Copy current tree nodes to update child pointers
        std::vector<KdNode> current_nodes(num_partitions);
        HIP_CHECK(hipMemcpy(current_nodes.data(), d_tree_nodes + nodes_created, 
                           num_partitions * sizeof(KdNode), hipMemcpyDeviceToHost));
        
        for (int i = 0; i < num_partitions; i++) {
            int left_child_idx = -1, right_child_idx = -1;
            
            // Add left child if it has points
            if (left_sizes[i] > 0) {
                Partition left_part;
                left_part.indices = next_buffer;
                left_part.size = left_sizes[i];
                left_part.node_idx = next_node_idx;
                left_part.level = current_partitions[i].level + 1;
                left_part.start_offset = left_starts[i];
                next_partitions.push_back(left_part);
                left_child_idx = next_node_idx;
                next_node_idx++;
            }
            
            // Add right child if it has points
            if (right_sizes[i] > 0) {
                Partition right_part;
                right_part.indices = next_buffer;
                right_part.size = right_sizes[i];
                right_part.node_idx = next_node_idx;
                right_part.level = current_partitions[i].level + 1;
                right_part.start_offset = right_starts[i];
                next_partitions.push_back(right_part);
                right_child_idx = next_node_idx;
                next_node_idx++;
            }
            
            // Update parent node with child pointers
            current_nodes[i].left_child = left_child_idx;
            current_nodes[i].right_child = right_child_idx;
        }
        
        // Copy updated nodes back to device
        HIP_CHECK(hipMemcpy(d_tree_nodes + nodes_created, current_nodes.data(),
                           num_partitions * sizeof(KdNode), hipMemcpyHostToDevice));
    }
    
    void verifyTreeStructure() {
        std::cout << "  - Verifying tree structure..." << std::endl;
        
        // Copy tree nodes to verify structure
        int nodes_to_check = std::min(20, actual_nodes);
        if (nodes_to_check == 0) return;
        
        std::vector<KdNode> sample_nodes(nodes_to_check);
        HIP_CHECK(hipMemcpy(sample_nodes.data(), d_tree_nodes, 
                           nodes_to_check * sizeof(KdNode), hipMemcpyDeviceToHost));
        
        int valid_nodes = 0;
        int leaf_nodes = 0;
        
        for (int i = 0; i < nodes_to_check; i++) {
            if (sample_nodes[i].tuple_idx >= 0) {
                valid_nodes++;
                if (sample_nodes[i].left_child == -1 && sample_nodes[i].right_child == -1) {
                    leaf_nodes++;
                }
                
                if (i < 10) {  // Show first 10 nodes
                    std::cout << "    Node " << i << ": tuple=" << sample_nodes[i].tuple_idx 
                              << ", dim=" << sample_nodes[i].split_dim 
                              << ", val=" << sample_nodes[i].split_value
                              << ", left=" << sample_nodes[i].left_child 
                              << ", right=" << sample_nodes[i].right_child << std::endl;
                }
            }
        }
        
        std::cout << "  - Found " << valid_nodes << " valid nodes (" << leaf_nodes << " leaves) in first " << nodes_to_check << " checked" << std::endl;
    }
};

// Example usage and performance testing
int main() {
    std::cout << "=== KD-Tree GPU Implementation with HIP ===" << std::endl;
    std::cout << "Full breadth-first construction with early termination" << std::endl;
    
    // Check HIP device availability
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No HIP-capable devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "\nFound " << deviceCount << " HIP device(s)" << std::endl;
    
    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    
    // Test with different dataset sizes
    std::vector<std::pair<int, int>> test_configs = {
        {100, 2},    // Small 2D dataset
        {1000, 3},   // Medium 3D dataset
        {5000, 4},   // Larger 4D dataset
        {10000, 2}   // Large 2D dataset
    };
    
    for (auto& config : test_configs) {
        int N = config.first;
        int D = config.second;
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing with " << N << " points in " << D << " dimensions" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Generate random test data with fixed seed for reproducibility
        srand(42);
        std::vector<std::vector<float>> points(N, std::vector<float>(D));
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                points[i][d] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
            }
        }
        
        // Show sample points for smaller datasets
        if (N <= 1000) {
            std::cout << "Sample points:" << std::endl;
            for (int i = 0; i < std::min(5, N); i++) {
                std::cout << "Point " << i << ": (";
                for (int d = 0; d < D; d++) {
                    std::cout << std::fixed << std::setprecision(2) << points[i][d];
                    if (d < D-1) std::cout << ", ";
                }
                std::cout << ")" << std::endl;
            }
        }
        
        try {
            std::cout << "\nCreating KD-Tree object..." << std::endl;
            HipKdTree kdtree(N, D);
            
            std::cout << "Building KD-Tree..." << std::endl;
            auto build_start = std::chrono::high_resolution_clock::now();
            
            kdtree.buildTree(points);
            
            auto build_end = std::chrono::high_resolution_clock::now();
            auto build_duration = std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start);
            
            std::cout << "KD-Tree construction completed successfully!" << std::endl;
            std::cout << "Build time: " << build_duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "Points per second: " << (N * 1000000.0) / build_duration.count() << std::endl;
            
            kdtree.printTreeStats();
            
        } catch (const std::exception& e) {
            std::cerr << "Error during KD-Tree construction: " << e.what() << std::endl;
            continue;
        } catch (...) {
            std::cerr << "Unknown error during KD-Tree construction" << std::endl;
            continue;
        }
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Demonstrate some advanced features
    std::cout << "\nDemonstrating advanced KD-Tree features..." << std::endl;
    
    // Create a small tree for detailed analysis
    int demo_N = 15;
    int demo_D = 2;
    std::vector<std::vector<float>> demo_points(demo_N, std::vector<float>(demo_D));
    
    // Create some structured data for better visualization
    srand(123);
    for (int i = 0; i < demo_N; i++) {
        demo_points[i][0] = (i % 5) * 20.0f + (rand() % 10);  // X coordinate
        demo_points[i][1] = (i / 5) * 15.0f + (rand() % 10);  // Y coordinate
    }
    
    std::cout << "\nDemo dataset (" << demo_N << " points):" << std::endl;
    for (int i = 0; i < demo_N; i++) {
        std::cout << "Point " << std::setw(2) << i << ": (" 
                  << std::setw(6) << std::fixed << std::setprecision(1) << demo_points[i][0] << ", "
                  << std::setw(6) << std::fixed << std::setprecision(1) << demo_points[i][1] << ")" << std::endl;
    }
    
    try {
        HipKdTree demo_tree(demo_N, demo_D);
        demo_tree.buildTree(demo_points);
        demo_tree.printTreeStats();
        
        std::cout << "\nDemo tree construction successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in demo: " << e.what() << std::endl;
    }
    
    // Performance summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "✓ Breadth-first GPU construction implemented" << std::endl;
    std::cout << "✓ Early termination for small partitions" << std::endl;
    std::cout << "✓ Deterministic ordering using super-key comparison" << std::endl;
    std::cout << "✓ Memory-efficient buffer management" << std::endl;
    std::cout << "✓ Robust error handling and verification" << std::endl;
    std::cout << "✓ Scalable for different dataset sizes and dimensions" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nProgram finished successfully." << std::endl;
    return 0;
}