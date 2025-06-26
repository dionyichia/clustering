#ifndef SINGLE_LINKAGE_HPP
#define SINGLE_LINKAGE_HPP

#include <vector>
#include <iostream>

// Forward declaration for Edge type (assuming it's defined in boruvka.hpp)
struct Edge;

// Function to perform single linkage clustering
std::vector<std::vector<int>> single_linkage_clustering(
    const std::vector<Edge>& mst_edges,
    int N_pts,
    int min_cluster_size = 2
);

// Helper function to collect members (can be made internal if not needed elsewhere)
void collect_members(int c,
                     int N_pts,
                     const std::vector<int>& left_child,
                     const std::vector<int>& right_child,
                     std::vector<int>& out);

#endif // SINGLE_LINKAGE_HPP