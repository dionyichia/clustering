#ifndef SINGLE_LINKAGE_HPP
#define SINGLE_LINKAGE_HPP

#include <vector>
#include <iostream>
#include "kd_tree/include/types.hpp"  // Include for Edge definition

// Structure to hold cluster selection choices
struct ClusterChoice {
    float total_stability;
    std::vector<int> selected_clusters;
    
    // default c-tor
    ClusterChoice() : total_stability(0.0f) {}

    // c-tor with arguments
    ClusterChoice(float stab, std::vector<int> clusters) 
        : total_stability(stab), selected_clusters(std::move(clusters)) {}
};

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