#ifndef BORUVKA_HPP
#define BORUVKA_HPP

#include <hip/hip_runtime.h>

// Type definitions (should match your boruvka_main.cpp)
typedef unsigned long long ullong;

struct __attribute__ ((packed)) Edge
{
    uint u; // Vertex index
    uint v;
    float weight;
    
    // Comparison operator for sorting
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct Vertex {
    ullong component;
    ullong cheapest_edge;
};

struct MST {
    char* mst; // Array indicating which edges are in MST
    double weight; // Total weight of MST
};

// Function declarations
MST boruvka_mst(const ullong n_vertices, const ullong n_edges, Edge* edge_list);

// Helper functions (if you want to expose them)
void print_mst_result(const MST& result, const Edge* edges, ullong n_edges);
void free_mst(MST& mst);

// TODO:: 

#endif // BORUVKA_HPP