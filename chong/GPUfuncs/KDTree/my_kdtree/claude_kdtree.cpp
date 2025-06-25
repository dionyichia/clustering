#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <queue>
using namespace cooperative_groups;

// Constants
#define WARP_SIZE 64  // AMD Wavefront size
#define MAX_THREADS_PER_BLOCK 256
#define MAX_GRID_SIZE 65535
#define HIP_CHECK(err)                                                      
  do {                                                                       
    hipError_t _e = (err);                                                   
    if (_e != hipSuccess) {                                                  
      std::cerr                                                             
        << "HIP error at " << __FILE__ << ":" << __LINE__                   
        << " → " << hipGetErrorString(_e) << std::endl;                     
      std::exit(EXIT_FAILURE);                                               
    }                                                                        
  } while (0)

// KD Tree Node structure
struct KdNode {
    int tuple_idx;        // Index into coordinate array
    int split_dim;        // Dimension to split on
    int left_child;       // Index of left child (-1 if leaf)
    int right_child;      // Index of right child (-1 if leaf)
    float split_value;    // Value at split dimension
};

// Partition structure for managing tree construction
struct Partition {
    int* indices;         // Array of point indices
    int size;            // Number of points in partition
    int node_idx;        // Tree node index for this partition
    int level;           // Tree level
};

// GPU Kernel for super key comparison
__device__ bool superKeyCompare(const float* coords, int D, int idx1, int idx2, int primary_dim) {
    // Compare primary dimension first
    float val1 = coords[idx1 * D + primary_dim];
    float val2 = coords[idx2 * D + primary_dim];
    
    if (val1 != val2) {
        return val1 < val2;
    }
    
    // If primary dimensions are equal, compare other dimensions
    for (int d = 0; d < D; d++) {
        if (d == primary_dim) continue;
        
        val1 = coords[idx1 * D + d];
        val2 = coords[idx2 * D + d];
        
        if (val1 != val2) {
            return val1 < val2;
        }
    }
    
    return false; // Equal tuples
}

// Quicksort partition kernel for finding medians
// 256 threads
__global__ void quicksortPartitionKernel(
    int* indices, 
    float* coords, 
    int D, 
    int dim,
    int* partition_starts, 
    int* partition_sizes,
    int* medians, 
    int num_partitions) {

    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;
    
    // finds indice of start of the partition in the flat indice array
    // flat indice array stores all point indices in one contiguous array
    int start = partition_starts[partition_id];
    // finds size of partition
    int size = partition_sizes[partition_id];
    // arr points to first element in this partition
    int* arr = indices + start;
    
    // if partition is only 1 member large, then its median is itself
    // trivial
    if (size <= 1) {
        if (size == 1) medians[partition_id] = arr[0];
        return;
    }
    
    // Use cooperative groups for warp-level operations
    // creates a thread_block object 
    thread_block block = this_thread_block();
    
    // Simple median-of-three pivot selection
    // Hoare Partition
    int pivot_idx = size / 2;
    int pivot = arr[pivot_idx];
    float pivot_val = coords[pivot * D + dim];
    
    // Partition array around pivot
    int left = 0, right = size - 1;
    
    while (left <= right) {
        // Find element on left that should be on right
        while (left < size && coords[arr[left] * D + dim] < pivot_val) {
            left++;
        }
        
        // Find element on right that should be on left
        while (right >= 0 && coords[arr[right] * D + dim] > pivot_val) {
            right--;
        }
        
        if (left <= right) {
            // Swap elements
            int temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }
    
    // Find actual median position
    // left = right + 1, since loop terminates when left > right
    int median_pos = size / 2;
    // if median lies in left of region
    // median should be largest of <= pivotVal, which is arr[right]
    if (median_pos <= right) {
        medians[partition_id] = arr[median_pos];
    // if median lies in right of region
    // median should be smallest of >= pivotVal, which is arr[left]
    } else {
        medians[partition_id] = arr[left];
    }
}

// Kernel to partition arrays based on medians
__global__ void partitionArraysKernel(int* input_indices, float* coords, int D, int dim,
                                     int* partition_starts, int* partition_sizes,
                                     int* medians, int* left_starts, int* right_starts,
                                     int* left_sizes, int* right_sizes,
                                     int* output_indices, int num_partitions) {
    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;
    
    int start = partition_starts[partition_id];
    int size = partition_sizes[partition_id];
    int median_idx = medians[partition_id];
    
    if (size <= 1) return;
    
    float split_val = coords[median_idx * D + dim];
    
    int left_count = 0, right_count = 0;
    int left_base = left_starts[partition_id];
    int right_base = right_starts[partition_id];
    
    // Partition points
    for (int i = 0; i < size; i++) {
        int point_idx = input_indices[start + i];
        if (point_idx == median_idx) continue;
        
        float val = coords[point_idx * D + dim];
        if (val < split_val || (val == split_val && 
            superKeyCompare(coords, D, point_idx, median_idx, dim))) {
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
    int median_idx = medians[tid];
    int split_dim = split_dims[tid];
    
    tree_nodes[node_idx].tuple_idx = median_idx;
    tree_nodes[node_idx].split_dim = split_dim;
    tree_nodes[node_idx].split_value = coords[median_idx * D + split_dim];
    tree_nodes[node_idx].left_child = -1;   // Will be set later
    tree_nodes[node_idx].right_child = -1;  // Will be set later
}

// Host class for KD-Tree construction
class HipKdTree {
private:
    float* d_coords;
    KdNode* d_tree_nodes;
    int* d_indices_buffer1;
    int* d_indices_buffer2;
    int N, D;
    int tree_size;
    
public:
    HipKdTree(int num_points, int dimensions) : N(num_points), D(dimensions) {
        std::cout << "Initializing HipKdTree with " << N << " points in " << D << " dimensions..." << std::endl;
        
        // Calculate tree size (upper bound)
        tree_size = 2 * N - 1;
        
        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_coords, N * D * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_tree_nodes, tree_size * sizeof(KdNode)));
        HIP_CHECK(hipMalloc(&d_indices_buffer1, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_indices_buffer2, N * sizeof(int)));
        
        std::cout << "Allocated memory: " << (N * D * sizeof(float) + tree_size * sizeof(KdNode) + 2 * N * sizeof(int)) / 1024.0 / 1024.0 << " MB" << std::endl;
    }
    
    ~HipKdTree() {
        hipFree(d_coords);
        hipFree(d_tree_nodes);
        hipFree(d_indices_buffer1);
        hipFree(d_indices_buffer2);
    }
    
    void buildTree(const std::vector<std::vector<float>>& points) {
        std::cout << "Starting tree construction..." << std::endl;
        
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
        HIP_CHECK(hipMemcpy(d_indices_buffer1, initial_indices.data(), N * sizeof(int), hipMemcpyHostToDevice));
        
        // Build tree using breadth-first approach
        buildTreeBreadthFirst();
        
        std::cout << "Tree construction completed." << std::endl;
    }
    
private:
    void buildTreeBreadthFirst() {
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
        current_partitions.push_back(root);
        
        int nodes_created = 0;
        int* current_buffer = d_indices_buffer1;
        int* next_buffer = d_indices_buffer2;
        
        while (!current_partitions.empty()) {
            int num_partitions = current_partitions.size();
            std::cout << "  - Processing level with " << num_partitions << " partitions..." << std::endl;
            
            if (num_partitions == 0) break;
            
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
            int current_offset = 0;
            for (int i = 0; i < num_partitions; i++) {
                partition_starts[i] = current_offset;
                partition_sizes[i] = current_partitions[i].size;
                node_indices[i] = current_partitions[i].node_idx;
                split_dims[i] = current_partitions[i].level % D;
                current_offset += current_partitions[i].size;
            }
            
            // Copy to device
            HIP_CHECK(hipMemcpy(d_partition_starts, partition_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_partition_sizes, partition_sizes.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_node_indices, node_indices.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_split_dims, split_dims.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            
            // Find medians
            dim3 gridSize(num_partitions);
            dim3 blockSize(256);
            
            hipLaunchKernelGGL(
                quicksortPartitionKernel,   // 1) The __global__ kernel function you’re invoking
                gridSize,                   // 2) <<<gridDim, blockDim, …>>> – number of thread-blocks
                blockSize,                  //     gridDim = gridSize (e.g. num_partitions blocks)
                0,                          // 3) Dynamic shared memory per block (bytes); here none
                0,                          // 4) HIP stream to launch on; 0 means the default stream
                /* then the kernel’s argument list */  
                current_buffer,             //   int* indices: pointer to this round’s index buffer  
                d_coords,                   //   float* coords: your coordinate array  
                D,                          //   int D: dimensionality  
                0,                          //   int split_dim: here fixed to 0 (first dimension)  
                d_partition_starts,         //   int* partition_starts: start offsets per partition  
                d_partition_sizes,          //   int* partition_sizes: sizes per partition  
                d_medians,                  //   int* medians: output medians per partition  
                num_partitions              //   int num_partitions: how many partitions to process
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            
            // Initialize tree nodes
            dim3 nodeGridSize((num_partitions + blockSize.x - 1) / blockSize.x);
            hipLaunchKernelGGL(initializeNodesKernel, nodeGridSize, blockSize, 0, 0,
                              d_tree_nodes, d_medians, d_coords, D,
                              d_node_indices, d_split_dims, num_partitions);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            
            // Prepare for partitioning
            std::vector<int> left_starts(num_partitions);
            std::vector<int> right_starts(num_partitions);
            current_offset = 0;
            
            for (int i = 0; i < num_partitions; i++) {
                left_starts[i] = current_offset;
                current_offset += partition_sizes[i] / 2 + 1;
                right_starts[i] = current_offset;
                current_offset += partition_sizes[i] / 2 + 1;
            }
            
            HIP_CHECK(hipMemcpy(d_left_starts, left_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_right_starts, right_starts.data(), num_partitions * sizeof(int), hipMemcpyHostToDevice));
            
            // Partition arrays
            hipLaunchKernelGGL(partitionArraysKernel, gridSize, blockSize, 0, 0,
                              current_buffer, d_coords, D, 0, // Use split dimension from split_dims
                              d_partition_starts, d_partition_sizes, d_medians,
                              d_left_starts, d_right_starts, d_left_sizes, d_right_sizes,
                              next_buffer, num_partitions);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            
            // Copy results back to host
            std::vector<int> left_sizes(num_partitions);
            std::vector<int> right_sizes(num_partitions);
            HIP_CHECK(hipMemcpy(left_sizes.data(), d_left_sizes, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(right_sizes.data(), d_right_sizes, num_partitions * sizeof(int), hipMemcpyDeviceToHost));
            
            // Prepare next level partitions
            next_partitions.clear();
            int next_offset = 0;
            
            for (int i = 0; i < num_partitions; i++) {
                // Add left child if it has points
                if (left_sizes[i] > 0) {
                    Partition left_part;
                    left_part.indices = next_buffer + next_offset;
                    left_part.size = left_sizes[i];
                    left_part.node_idx = nodes_created + next_partitions.size() * 2 + 1;
                    left_part.level = current_partitions[i].level + 1;
                    next_partitions.push_back(left_part);
                    next_offset += left_sizes[i];
                }
                
                // Add right child if it has points
                if (right_sizes[i] > 0) {
                    Partition right_part;
                    right_part.indices = next_buffer + next_offset;
                    right_part.size = right_sizes[i];
                    right_part.node_idx = nodes_created + next_partitions.size() * 2 + 2;
                    right_part.level = current_partitions[i].level + 1;
                    next_partitions.push_back(right_part);
                    next_offset += right_sizes[i];
                }
            }
            
            nodes_created += num_partitions;
            
            // Clean up device memory for this level
            hipFree(d_partition_starts);
            hipFree(d_partition_sizes);
            hipFree(d_medians);
            hipFree(d_node_indices);
            hipFree(d_split_dims);
            hipFree(d_left_starts);
            hipFree(d_right_starts);
            hipFree(d_left_sizes);
            hipFree(d_right_sizes);
            
            // Swap buffers and partitions
            std::swap(current_buffer, next_buffer);
            current_partitions = std::move(next_partitions);
            
            std::cout << "    Created " << num_partitions << " nodes, next level has " << current_partitions.size() << " partitions" << std::endl;
        }
        
        std::cout << "  - Tree construction completed with " << nodes_created << " nodes" << std::endl;
    }
};

// Example usage
int main() {
    std::cout << "Starting KD-Tree GPU implementation with full breadth-first construction..." << std::endl;
    
    // Check HIP device availability
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No HIP-capable devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;
    
    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Example: Build KD-Tree for 1000 3D points
    int N = 1000;
    int D = 3;
    
    std::cout << "Generating " << N << " random " << D << "D points..." << std::endl;
    
    // Generate random test data
    srand(42); // Fixed seed for reproducible results
    std::vector<std::vector<float>> points(N, std::vector<float>(D));
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            points[i][d] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        }
    }
    
    std::cout << "Sample points:" << std::endl;
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "Point " << i << ": (";
        for (int d = 0; d < D; d++) {
            std::cout << points[i][d];
            if (d < D-1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    try {
        std::cout << "Creating KD-Tree object..." << std::endl;
        HipKdTree kdtree(N, D);
        
        std::cout << "Building KD-Tree..." << std::endl;
        kdtree.buildTree(points);
        
        std::cout << "KD-Tree construction completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during KD-Tree construction: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Program finished successfully." << std::endl;
    return 0;
}