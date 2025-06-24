#include <hip/hip_runtime.h>
#include <string>
#include <cstdio>  // for printf
#include <climits> // for ULONG_MAX
#include <iostream>

// Add error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

typedef unsigned long long ullong;

// Define constants for HIP threadblocks
#define BLOCKSIZE_EDGE (256)        // 4 wavefronts for edge-heavy kernels
#define BLOCKSIZE_VERTEX (128)      // 2 wavefronts for vertex-heavy kernels  
#define NBLOCKS_ASSIGN_CHEAPEST (384)  // 384 blocks × 4 waves = 1,536 waves (87% occupancy)
#define NBLOCKS_OTHER (88)          // 88 blocks × 2 waves = 176 waves per vertex kernel

// For compatibility:
#define BLOCKSIZE BLOCKSIZE_EDGE

#define NTHREADS_ASSIGN_CHEAPEST (NBLOCKS_ASSIGN_CHEAPEST * BLOCKSIZE_EDGE)  // 98,304 threads
#define NTHREADS_OTHER (NBLOCKS_OTHER * BLOCKSIZE_VERTEX)                    // 11,264 threads

#define NO_EDGE (ULONG_MAX)
 
struct __attribute__ ((packed)) Edge
{
    uint u; // Vertice index will probably not exceed 4 billion
    uint v;
    uint weight;
};

struct Vertex {
    ullong component;
    ullong cheapest_edge; // Edge indexes might exceed 4 billion, ( if graph is fully connected with > 100,000 vertices) therefore ullong
};

struct MST {
    char * mst; // Changed to only store selected edge indicies
    ullong weight;
};

// This stores the global constants
struct GlobalConstants {
    Vertex* vertices;
    Edge* edges;
    ullong* mst;
    ullong n_vertices;
    ullong n_edges;
};

// Another global value
__device__ ullong n_unions_total;
__device__ ullong mst_weight_total;

// Global variable that is in scope, but read-only, for all hip
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU.
__constant__ GlobalConstants hipConstGraphParams;

// Device function declarations
__device__ inline int edge_cmp(const Edge* edges, const ullong i, const ullong j);
__device__ inline ullong get_component(Vertex* componentlist, const ullong i);
__device__ inline void flatten_component(Vertex* componentlist, const ullong i);
__device__ inline void merge_components(Vertex* componentlist, const ullong i, const ullong j);
__device__ inline bool safe_merge_components(Vertex* componentlist, const ullong i, const ullong j);

// Kernel declarations
__global__ void init_arrs();
__global__ void reset_arrs();
__global__ void assign_cheapest();
__global__ void update_mst();
__global__ void update_mst_phase1();
__global__ void update_mst_phase2();
__global__ void update_mst_simple();

// Add debug kernel to print device state
__global__ void debug_print_state(const char* label, int max_print = 20) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== %s ===\n", label);
        printf("n_vertices: %llu, n_edges: %llu\n", 
               hipConstGraphParams.n_vertices, hipConstGraphParams.n_edges);
        printf("n_unions_total: %llu, mst_weight_total: %llu\n", 
               n_unions_total, mst_weight_total);
        
        // Print first few vertices
        printf("Vertices (first %d):\n", max_print);
        for (int i = 0; i < min(max_print, (int)hipConstGraphParams.n_vertices); i++) {
            printf("  v[%d]: component=%llu, cheapest_edge=%llu\n", 
                   i, hipConstGraphParams.vertices[i].component, 
                   hipConstGraphParams.vertices[i].cheapest_edge);
        }
        
        // Print first few edges
        printf("Edges (first %d):\n", max_print);
        for (int i = 0; i < min(max_print, (int)hipConstGraphParams.n_edges); i++) {
            printf("  e[%d]: u=%u, v=%u, weight=%u\n", 
                   i, hipConstGraphParams.edges[i].u, 
                   hipConstGraphParams.edges[i].v, 
                   hipConstGraphParams.edges[i].weight);
        }
        
        // Print MST array
        printf("MST array (first %d):\n", max_print);
        for (int i = 0; i < min(max_print, (int)hipConstGraphParams.n_vertices); i++) {
            printf("  mst[%d]: %llu\n", i, hipConstGraphParams.mst[i]);
        }
    }
}


MST boruvka_mst(const ullong n_vertices, const ullong n_edges, Edge* edge_list) {
    MST mst;
    mst.weight = 0;

    // DEBUG 1: Print input validation
    printf("=== INPUT VALIDATION ===\n");
    printf("n_vertices: %llu, n_edges: %llu\n", n_vertices, n_edges);
    if (n_edges == 0 || n_vertices == 0) {
        printf("ERROR: Invalid input - zero vertices or edges\n");
        return mst;
    }
    
    // Print first few input edges
    printf("Input edges (first 5):\n");
    for (int i = 0; i < min(5, (int)n_edges); i++) {
        printf("  edge[%d]: u=%u, v=%u, weight=%u\n", 
               i, edge_list[i].u, edge_list[i].v, edge_list[i].weight);
    }

    std::cout << "start malloc : " << std::endl;

    ullong* mst_tree = (ullong *) malloc((n_vertices) * sizeof(ullong));
    
    // DEBUG 2: Initialize host array for comparison
    for (ullong i = 0; i < n_vertices; i++) {
        mst_tree[i] = NO_EDGE;
    }

    ullong* d_mst;
    Vertex* d_vertices;
    Edge* d_edges;

    // Add error checking to all HIP calls
    HIP_CHECK(hipMalloc(&d_mst, sizeof(ullong) * n_vertices));
    HIP_CHECK(hipMemset(d_mst, 0, sizeof(ullong) * n_vertices));
    HIP_CHECK(hipMalloc(&d_vertices, sizeof(Vertex) * n_vertices));
    HIP_CHECK(hipMalloc(&d_edges, sizeof(Edge) * n_edges));
    HIP_CHECK(hipMemcpy(d_edges, edge_list, sizeof(Edge) * n_edges, hipMemcpyHostToDevice));

    GlobalConstants params;
    params.vertices = d_vertices;
    params.edges = d_edges;
    params.mst = d_mst;
    params.n_edges = n_edges;
    params.n_vertices = n_vertices;

    HIP_CHECK(hipMemcpyToSymbol(hipConstGraphParams, &params, sizeof(GlobalConstants)));

    ullong n_unions = 0;
    ullong mst_weight = 0;
    ullong n_unions_old;

    HIP_CHECK(hipMemcpyToSymbol(n_unions_total, &n_unions, sizeof(ullong)));
    HIP_CHECK(hipMemcpyToSymbol(mst_weight_total, &mst_weight, sizeof(ullong)));

    // DEBUG 3: Check kernel launch parameters
    printf("=== KERNEL LAUNCH PARAMS ===\n");
    printf("init_arrs: %d blocks, %d threads per block\n", NBLOCKS_OTHER, BLOCKSIZE_VERTEX);
    printf("assign_cheapest: %d blocks, %d threads per block\n", NBLOCKS_ASSIGN_CHEAPEST, BLOCKSIZE_EDGE);

    init_arrs<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
    HIP_CHECK(hipDeviceSynchronize());

    // DEBUG 4: Check initialization
    debug_print_state<<<1, 1>>>("AFTER INIT");
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "end malloc : " << std::endl;

    int iteration = 0;
    do {
        printf("\n=== ITERATION %d ===\n", iteration++);
        n_unions_old = n_unions;

        std::cout << "resetting: " << std::endl;
        reset_arrs<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
        HIP_CHECK(hipDeviceSynchronize());

        // DEBUG 5: Check after reset
        debug_print_state<<<1, 1>>>("AFTER RESET");
        HIP_CHECK(hipDeviceSynchronize());

        std::cout << "assigning cheapest: " << std::endl;
        assign_cheapest<<<NBLOCKS_ASSIGN_CHEAPEST, BLOCKSIZE_EDGE>>>();
        HIP_CHECK(hipDeviceSynchronize());

        // DEBUG 6: Check after assign_cheapest
        debug_print_state<<<1, 1>>>("AFTER ASSIGN_CHEAPEST");
        HIP_CHECK(hipDeviceSynchronize());

        std::cout << "updating: " << std::endl;
        // update_mst<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
        // update_mst_phase1<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
        // update_mst_phase2<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
        update_mst_simple<<<NBLOCKS_OTHER, BLOCKSIZE_VERTEX>>>();
        HIP_CHECK(hipDeviceSynchronize());

        // DEBUG 7: Check after update_mst
        debug_print_state<<<1, 1>>>("AFTER UPDATE_MST");
        HIP_CHECK(hipDeviceSynchronize());

        // Copy back current state
        HIP_CHECK(hipMemcpyFromSymbol(&n_unions, n_unions_total, sizeof(ullong)));
        HIP_CHECK(hipMemcpyFromSymbol(&mst_weight, mst_weight_total, sizeof(ullong)));

        printf("Iteration %d: n_unions_old=%llu, n_unions=%llu, mst_weight=%llu\n", 
               iteration-1, n_unions_old, n_unions, mst_weight);

        // DEBUG 8: Prevent infinite loops
        if (iteration > n_vertices) {
            printf("ERROR: Too many iterations, breaking\n");
            break;
        }

    } while (n_unions != n_unions_old && n_unions < n_vertices - 1);

    std::cout << "gpu loop end copying back to host: " << std::endl;

    // DEBUG 9: Final state check
    debug_print_state<<<1, 1>>>("FINAL STATE");
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(mst_tree, d_mst, sizeof(ullong) * n_vertices, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbol(&mst_weight, mst_weight_total, sizeof(ullong)));

    // DEBUG 10: Print final results
    printf("\n=== FINAL RESULTS ===\n");
    printf("Final MST weight: %llu\n", mst_weight);
    printf("Final n_unions: %llu (expected: %llu)\n", n_unions, n_vertices - 1);
    printf("MST edges:\n");
    for (ullong i = 0; i < n_vertices; i++) {
        if (mst_tree[i] != NO_EDGE) {
            printf("  mst[%llu] = edge %llu\n", i, mst_tree[i]);
        }
    }

    mst.mst = mst_tree;
    mst.weight = mst_weight;

    std::cout << "done copying" << std::endl;

    for (ullong i = 0; i < n_vertices; i++) {
        if (mst.mst[i] != NO_EDGE) {
            printf("  mst[%llu] = edge %llu\n", i, mst.mst[i]);
        }
    }


    hipFree(d_mst);
    hipFree(d_vertices);
    hipFree(d_edges);

    return mst;
}

__device__ inline int edge_cmp(const Edge* edges, const ullong i, const ullong j)
{
    if (i == j) return 0;

    const Edge& lhs = edges[i];
    const Edge& rhs = edges[j];

    if (lhs.weight < rhs.weight) {
        return -1;
    }
    else if (lhs.weight > rhs.weight) {
        return 1;
    }
    // If same weight pick the one with smaller idx
    else if (i < j) {
        return -1;
    } 
    else {
        return 1;
    }
}

// Cannot do path compression, here, multiple threads will acess this at the same time
__device__ inline ullong get_component(Vertex* componentlist, const ullong i) {
    // ullong curr = componentlist[i].component;
    // while (componentlist[curr].component != curr) {
    //     curr = componentlist[curr].component;
    // }
    // return curr;

    ullong curr = i;
    ullong next;
    
    // Single traversal to root with volatile reads
    while ((next = ((volatile Vertex*)componentlist)[curr].component) != curr) {
        curr = next;
    }
    return curr;
}

// Merge flatten in merge_comp
__device__ inline void flatten_component(Vertex* componentlist, const ullong i) {
    ullong parent = componentlist[i].component;

    while (componentlist[parent].component != parent) {
        parent = componentlist[parent].component;
    }

    // Flatten component trees
    componentlist[i].component = parent;
    // if (componentlist[i].component != curr) {
    //     atomicExch(&componentlist[i].component, curr);
    // }
}

// TODO: Check atomicCAS
__device__ inline void merge_components(Vertex* componentlist, const ullong i,
                                        const ullong j) {
    // ullong u = i;
    // ullong v = j;
    // componentlist[i].component = j;
    // const ullong v = get_component(componentlist, j);
    // ullong old;
    // do {
    //     u = get_component(componentlist, u);
    //     old = atomicCAS(&(componentlist[u].component), u, v);
    // } while (old != u);
        // Find roots of both components
    ullong root_i = get_component(componentlist, i);
    ullong root_j = get_component(componentlist, j);
    
    if (root_i == root_j) return; // Already merged
    
    // Ensure consistent ordering to prevent deadlocks
    if (root_i > root_j) {
        ullong temp = root_i; root_i = root_j; root_j = temp;
    }
    
    // Single atomic operation - if it fails, another thread succeeded
    ullong old = atomicCAS((unsigned long long*)&componentlist[root_j].component, root_j, root_i);
    return;
}

__device__ inline bool safe_merge_components(Vertex* componentlist, const ullong i, const ullong j) {
    const int MAX_RETRIES = 10;
    
    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        // Get current roots
        ullong root_i = get_component(componentlist, i);
        ullong root_j = get_component(componentlist, j);
        
        if (root_i == root_j) {
            return false; // Already merged by another thread
        }
        
        // Ensure consistent ordering
        if (root_i > root_j) {
            ullong temp = root_i; root_i = root_j; root_j = temp;
        }
        
        // Try to merge - if successful, we're done
        ullong old_val = atomicCAS((unsigned long long*)&componentlist[root_j].component, root_j, root_i);
        
        if (old_val == root_j) {
            return true; // Success!
        }
        
        // If failed, the root might have changed, retry
        // Add small delay to reduce contention
        if (retry > 5) {
            __threadfence();
        }
    }
    
    return false; // Failed after retries
}

__global__ void init_arrs() {
    // const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // const int blockID = blockIdx.x;
    // const int block_width = blockDim.x;

    // const ullong n_vertices = hipConstGraphParams.n_vertices;
    // Vertex* const vertices = hipConstGraphParams.vertices;

    // const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    // const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    // // initialize components
    // for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
    //     vertices[i] = Vertex{i, NO_EDGE};
    //     hipConstGraphParams.mst[i] = NO_EDGE;

    // }

    const ullong threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const ullong n_vertices = hipConstGraphParams.n_vertices;
    const ullong total_threads = gridDim.x * blockDim.x;
    
    // Each thread handles multiple vertices to improve memory throughput
    for (ullong i = threadID; i < n_vertices; i += total_threads) {
        hipConstGraphParams.vertices[i] = Vertex{i, NO_EDGE};
        hipConstGraphParams.mst[i] = NO_EDGE;
    }
}

__global__ void reset_arrs() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    const ullong n_vertices = hipConstGraphParams.n_vertices;
    Vertex* const vertices = hipConstGraphParams.vertices;

    const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    // initialize components
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        vertices[i].cheapest_edge = NO_EDGE;
        flatten_component(vertices, i);
    }
}

__global__ void assign_cheapest() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    // Renaming to make life easier, this gets compiled away, loads from read-only const memory
    const ullong n_edges = hipConstGraphParams.n_edges;
    Vertex* const vertices = hipConstGraphParams.vertices;
    Edge* const edges = hipConstGraphParams.edges;

    const ullong block_start = (blockID * n_edges) / NBLOCKS_ASSIGN_CHEAPEST;
    const ullong block_end = ((blockID + 1) * n_edges) / NBLOCKS_ASSIGN_CHEAPEST;

    // Interleave acccess to list to take advantage of SIMD execution within a warp
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        // __syncthreads(); test removing this, i dont think it is required as no shared memory between threads
        Edge e = edges[i];

        // get root of component u, CHANGLOG: i changed it to assigning root_u to a new variable instead of the edge
        // global params is read-only memory anyways, &  memory access will be slower
        // Initial: e.u = get_component(vertices, e.u);
        ullong root_u = get_component(vertices, e.u);

        // get root of component v
        ullong root_v = get_component(vertices, e.v);

        // Skip edges that connect a component to itself
        if (root_u == root_v) {
            continue;
        }

        // Atomic update cheapest_edge[u]
        // Lock-free update pattern
        // This is the inital cheapeast edge of vertex
        ullong expected = vertices[root_u].cheapest_edge;
        // This is your current candidate cheapest edge to be compared with the initial above
        ullong old;
        // While loop condition, if curr candidate still less than initial, execute
        while (expected == NO_EDGE || edge_cmp(edges, i, expected) < 0) {
            // try to update the cheapest edge, return the updated value (candidate edge)
            old = atomicCAS(&vertices[root_u].cheapest_edge, expected, i);

            // If the cheapest edge is now ur candidate value, break
            if (expected == old) {
                break;
            }

            // Else update the initial cheapest edge of vertex to the new value. 
            // (new value beacuse some other thread updated it first, since atomicCAS allows 1 thread to update while the rest wait)
            expected = old;
        }

        // Atomic update cheapest_edge[v]
        expected = vertices[root_v].cheapest_edge;
        while (expected == NO_EDGE || edge_cmp(edges, i, expected) < 0) {
            old = atomicCAS(&vertices[root_v].cheapest_edge, expected, i);
            if (expected == old) {
                break;
            }
            expected = old;
        }
    }
}

__global__ void update_mst() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    // Renaming to make life easier, this gets compiled away
    const ullong n_vertices = hipConstGraphParams.n_vertices;
    Vertex* const vertices = hipConstGraphParams.vertices;
    Edge* const edges = hipConstGraphParams.edges;

    const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    ullong n_unions_made = 0;
    ullong batch_edge_weight = 0;

    // Connect newest edges to MST
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        const ullong edge_ind = vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edges[edge_ind];

        // If a cheapest edge connects two different components A and B, both endpoints of that edge will see it as their cheapest_edge.
        // Now in update_mst(), both vertex 3 and 7 will hit edge 12.
            // At i = 3, we detect that:
                // edge_ptr.u == i (3 == 3)
            // and edge_ind == vertices[edge_ptr.v].cheapest_edge (edge 12 is still the cheapest for 7)
            // → So skip it.
            // At i = 7, edge_ptr.u = 7, edge_ptr.v = 7 → do the merge.
        if (edge_ptr.u == i &&
            edge_ind == vertices[edge_ptr.v].cheapest_edge) {
            continue;
        }

        const ullong j = (i == edge_ptr.u? edge_ptr.v : edge_ptr.u); // this is the other index

        // Race Conditions not possible because 1 edge will have max 2 parent components, 
        // and higher idx parent is skipped with earlier condition
        // store the selected edge index at the vertice indx of the ullong array, 
        // since this is parralised by vertice, no race conditions
        hipConstGraphParams.mst[i] = edge_ind;

        // sum weights for all cheapest edges in vertice batch, 
        // no double counting as earlier condition would have skipped higher idx parent.
        batch_edge_weight += edge_ptr.weight;

        // Debug
        ullong root_i_before = get_component(vertices, i);
        ullong root_j_before = get_component(vertices, j);
        merge_components(vertices, i, j);
        ullong root_i_after = get_component(vertices, i);
        ullong root_j_after = get_component(vertices, j);

        if (threadID < 10) {
            printf("Thread %d: Before merge: %llu->%llu, %llu->%llu. After: %llu->%llu, %llu->%llu, MST_weight_total: %llu, Batch edge weight: %llu\n",
                threadID, i, root_i_before, j, root_j_before, i, root_i_after, j, root_j_after,
                mst_weight_total, batch_edge_weight);
        }
        n_unions_made++;
    }

    atomicAdd(&mst_weight_total, batch_edge_weight);
    atomicAdd(&n_unions_total, n_unions_made);
}

// Phase 1: Mark edges for inclusion (no actual merging)
// Phase 2: Perform merges sequentially or with better synchronization
__global__ void update_mst_phase1() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    const ullong n_vertices = hipConstGraphParams.n_vertices;
    Vertex* const vertices = hipConstGraphParams.vertices;
    Edge* const edges = hipConstGraphParams.edges;

    const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    // Phase 1: Only validate and mark edges, no merging yet
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        const ullong edge_ind = vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edges[edge_ind];

        // Skip duplicate processing
        if (edge_ptr.u == i && edge_ind == vertices[edge_ptr.v].cheapest_edge) {
            continue;
        }

        const ullong j = (i == edge_ptr.u ? edge_ptr.v : edge_ptr.u);

        // Get current roots (these might change, but we'll validate later)
        ullong root_i = get_component(vertices, i);
        ullong root_j = get_component(vertices, j);

        // Only proceed if components are different, if same then remove the selected edge
        if (root_i != root_j) {
            atomicCAS(&hipConstGraphParams.mst[i], NO_EDGE, edge_ind);
        }
    }
}

__global__ void update_mst_phase2() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    const ullong n_vertices = hipConstGraphParams.n_vertices;
    Vertex* const vertices = hipConstGraphParams.vertices;
    Edge* const edges = hipConstGraphParams.edges;

    const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    ullong n_unions_made = 0;
    ullong batch_edge_weight = 0;

    // Phase 2: Perform actual merges with validation
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        const ullong edge_ind = hipConstGraphParams.mst[i];

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edges[edge_ind];
        const ullong j = (i == edge_ptr.u ? edge_ptr.v : edge_ptr.u);

        // Double-check that this edge is still valid
        if (safe_merge_components(vertices, i, j)) {
            batch_edge_weight += edge_ptr.weight;
            n_unions_made++;
        }
    }

    atomicAdd(&mst_weight_total, batch_edge_weight);
    atomicAdd(&n_unions_total, n_unions_made);
}

__global__ void update_mst_simple() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockID = blockIdx.x;
    const int block_width = blockDim.x;

    const ullong n_vertices = hipConstGraphParams.n_vertices;
    Vertex* const vertices = hipConstGraphParams.vertices;
    Edge* const edges = hipConstGraphParams.edges;

    const ullong block_start = (blockID * n_vertices) / NBLOCKS_OTHER;
    const ullong block_end = ((blockID + 1) * n_vertices) / NBLOCKS_OTHER;

    // Process vertices directly
    for (ullong i = block_start + threadIdx.x; i < block_end; i += block_width) {
        const ullong edge_ind = vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edges[edge_ind];

        // This find vertice j, the other vertice connected to the edge
        const ullong j = (i == edge_ptr.u ? edge_ptr.v : edge_ptr.u);

        bool is_shared = (edge_ind == vertices[j].cheapest_edge);
        if (is_shared && i < j) {
            continue;
        }

        // Skip duplicate processing
        // if (edge_ptr.u == i && edge_ind == vertices[edge_ptr.v].cheapest_edge) {
        //     continue;
        // }

        // Get current roots
        ullong root_i = get_component(vertices, i);
        ullong root_j = get_component(vertices, j);

        // Process if components are different
        if (root_i != root_j) {
            // Try to merge components
            if (safe_merge_components(vertices, i, j)) {
                // Store in MST array
                hipConstGraphParams.mst[i] = edge_ind;
                
                // Direct atomic updates (simpler but potentially slower)
                atomicAdd(&mst_weight_total, (ullong)edge_ptr.weight);
                atomicAdd(&n_unions_total, 1ULL);
            }
        }
    }
}


void initGPUs() {
    int deviceCount = 0;
    // bool isFastGPU = false;
    std::string name;
    hipError_t err = hipGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing HIP for Parallel Boruvka's\n");
    printf("Found %d HIP devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t  deviceProps;
        hipGetDeviceProperties(&deviceProps, i);

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   HIP Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }

    printf("---------------------------------------------------------\n");
}




