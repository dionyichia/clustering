#ifndef BORUVKA_HPP
#define BORUVKA_HPP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>      // for printf
#include <cstdlib>     // for exit
#include <climits>     // for ULONG_MAX

// Error checking macro
#ifndef HIP_CHECK
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            printf("HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)
#endif

// Type definitions
using ullong = unsigned long long;

// HIP thread- and block-size constants
#define BLOCKSIZE_EDGE            (256)
#define BLOCKSIZE_VERTEX          (128)
#define NBLOCKS_ASSIGN_CHEAPEST   (384)
#define NBLOCKS_OTHER             (88)
#define NTHREADS_ASSIGN_CHEAPEST  (NBLOCKS_ASSIGN_CHEAPEST * BLOCKSIZE_EDGE)
#define NTHREADS_OTHER            (NBLOCKS_OTHER         * BLOCKSIZE_VERTEX)
#define NO_EDGE                   (ULONG_MAX)

// Graph data structures
// struct __attribute__((packed)) Edge {
//     uint32_t u;    // endpoint index
//     uint32_t v;    // endpoint index
//     float    weight;
// };

struct Vertex {
    ullong component;
    ullong cheapest_edge;
};

struct MST {
    char*   mst;    // selection flags per edge
    double  weight; // total MST weight
};

// Global constants stored in __constant__ memory on device
struct GlobalConstants {
    Vertex* vertices;
    Edge*   edges;
    char*   mst;
    ullong  n_vertices;
    ullong  n_edges;
};

// Extern device globals
extern __constant__ GlobalConstants hipConstGraphParams;
extern __device__ ullong     n_unions_total;
extern __device__ double     mst_weight_total;

// Device helper functions
__device__ int    edge_cmp(const Edge* edges, ullong i, ullong j);
__device__ ullong get_component(Vertex* componentlist, ullong i);
__device__ void   flatten_component(Vertex* componentlist, ullong i);
__device__ void   merge_components(Vertex* componentlist, ullong i, ullong j);
__device__ bool   safe_merge_components(Vertex* componentlist, ullong i, ullong j);

// Kernel declarations
__global__ void init_arrs();
__global__ void reset_arrs();
__global__ void assign_cheapest();
__global__ void update_mst();
__global__ void update_mst_simple();
__global__ void debug_print_state(const char* label, int max_print = 20);

// Host API
// Computes MST using Boruvka's algorithm on GPU
MST boruvka_mst(ullong n_vertices, ullong n_edges, Edge* edge_list);

// Initializes available HIP devices (printing device info)
void initGPUs();

#endif // BORUVKA_HPP
