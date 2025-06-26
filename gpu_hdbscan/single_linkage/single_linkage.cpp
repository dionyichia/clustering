#include "single_linkage.hpp"
#include "kd_tree/include/types.hpp"  // Include for Edge definition
#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>

// Recursively gather all original points under cluster `c`.
void collect_members(int c,
                     int N_pts,
                     const std::vector<int>& left_child,
                     const std::vector<int>& right_child,
                     std::vector<int>& out)
{
    if (c < N_pts) {
        // leaf
        out.push_back(c);
    } else {
        collect_members(left_child[c], N_pts, left_child, right_child, out);
        collect_members(right_child[c], N_pts, left_child, right_child, out);
    }
}

std::vector<std::vector<int>> single_linkage_clustering(
    const std::vector<Edge>& mst_edges,
    int N_pts,
    int min_cluster_size)
{
    std::cout << "\n=== Running Single Linkage Clustering ===" << std::endl;

    // Make a copy of mst_edges for sorting
    std::vector<Edge> edges_copy = mst_edges;
    
    /* 
        COPY OUTPUT FROM BORUVKA INTO GPU AND RUN THRUST SORT, SHOULD BE FASTER
        BUT WILL INCUR COPY FROM HOST TO DEVICE COST
    */
    // assume `mst_edges` is your input vector<Edge> of size N–1
    std::cout << "[DEBUG] Before sort, first few weights:";
    for (int i = 0; i < std::min<size_t>(5, edges_copy.size()); ++i)
        std::cout << " " << edges_copy[i].weight;
    std::cout << "\n";

    // Sort edges by weight
    std::sort(edges_copy.begin(), edges_copy.end(),
              [](const Edge &a, const Edge &b) { return a.weight < b.weight; });

    std::cout << "[DEBUG] After sort, smallest 5 weights:";
    for (int i = 0; i < std::min<size_t>(5, edges_copy.size()); ++i)
        std::cout << " " << edges_copy[i].weight;
    std::cout << "\n";

    // After sorting:
    assert(!edges_copy.empty());
    for (auto &e : edges_copy) {
        assert(e.u >= 0 && e.u < N_pts);
        assert(e.v >= 0 && e.v < N_pts);
        assert(e.weight > 0);
    }
    std::cout << "[DEBUG] Edge assertions passed.\n";
    
    int max_clusters = 2 * N_pts;
    std::vector<int> parent(max_clusters), sz(max_clusters);
    std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // Guard against zero-length edges
    float smallest_weight = edges_copy.front().weight;
    std::cout << "[DEBUG] Raw smallest weight: " << smallest_weight << "\n";
    if (smallest_weight <= 0.f) {
        smallest_weight = std::numeric_limits<float>::min();
        std::cout << "[DEBUG] Adjusted smallest weight to epsilon: " << smallest_weight << "\n";
    }

    // Compute lambda_max as the inverse of smallest_mrd
    float lambda_max = 1.f / smallest_weight;
    std::cout << "[DEBUG] lambda_max: " << lambda_max << "\n";
    
    /* initialise all points as singleton clusters */
    for(int i = 0; i < N_pts; ++i){
        parent[i] = i;
        sz[i] = 1;
        birth_lambda[i] = lambda_max;
        death_lambda[i] = 0;
        stability[i] = 0;
    }
    int next_cluster_id = N_pts;
    std::cout << "[DEBUG] Initialized " << next_cluster_id << " singleton clusters\n";

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

    // Build hierarchy
    for(auto &e : edges_copy){
        int c1 = find_root(e.u), c2 = find_root(e.v);
        if(c1 == c2) continue;

        float lambda = 1.f / e.weight;
        death_lambda[c1] = death_lambda[c2] = lambda;
        stability[c1] += (birth_lambda[c1] - lambda) * sz[c1];
        stability[c2] += (birth_lambda[c2] - lambda) * sz[c2];

        int c_new = next_cluster_id++;
        parent[c1] = parent[c2] = c_new;
        parent[c_new] = c_new;
        sz[c_new]           = sz[c1] + sz[c2];
        birth_lambda[c_new] = lambda;
        stability[c_new]    = 0;
        death_lambda[c_new] = 0;
        left_child[c_new]   = c1;
        right_child[c_new]  = c2;

        std::cout << "[DEBUG] Merged clusters " << c1 << " and " << c2
                << " into " << c_new << " at lambda=" << lambda << "\n";
    }

    std::cout << "[DEBUG] Total clusters created: " << next_cluster_id << "\n";

    // Finalize singleton deaths
    for(int c = 0; c < next_cluster_id; ++c){
        if(parent[c] == c){
            death_lambda[c] = 0;
            stability[c]   += (birth_lambda[c] - 0) * sz[c];
        }
    }

    // Collect candidates
    std::vector<int> candidates;
    for(int c = 0; c < next_cluster_id; ++c){
        if(sz[c] >= min_cluster_size && death_lambda[c] > 0)
            candidates.push_back(c);
    }
    std::cout << "[DEBUG] Number of candidate clusters (size>=" << min_cluster_size
            << "): " << candidates.size() << "\n";

    // Sort candidates by stability
    std::sort(candidates.begin(), candidates.end(),
        [&](int a, int b){ return stability[a] > stability[b]; });
    std::cout << "[DEBUG] Top 5 candidate stabilities:";
    for (int i = 0; i < std::min<int>(5, candidates.size()); ++i)
        std::cout << " (" << candidates[i] << ":" << stability[candidates[i]] << ")";
    std::cout << "\n";

    // Select final clusters
    std::vector<bool> is_selected(max_clusters, false);
    std::vector<int> final_clusters;
    for(int c : candidates) {
        int L = left_child[c], R = right_child[c];
        if ((L >= N_pts && is_selected[L]) || (R >= N_pts && is_selected[R])) {
            std::cout << "[DEBUG] Skipping cluster " << c << " because child already selected\n";
            continue;
        }
        is_selected[c] = true;
        final_clusters.push_back(c);
        std::cout << "[DEBUG] Selected cluster " << c << "\n";
    }
    std::cout << "[DEBUG] Total final clusters: " << final_clusters.size() << "\n";

    // Assign points to clusters
    std::vector<int> assignment(N_pts, -1);
    std::vector<std::vector<int>> clusters;
    
    for(int c : final_clusters){
        std::vector<int> mem;
        collect_members(c, N_pts, left_child, right_child, mem);
        std::vector<int> this_cluster;
        for(int p : mem){
            if(assignment[p] == -1){
                assignment[p] = clusters.size();
                this_cluster.push_back(p);
            }
        }
        if(!this_cluster.empty()){
            clusters.push_back(std::move(this_cluster));
            std::cout << "[DEBUG] Cluster " << (clusters.size()-1)
                    << " got " << clusters.back().size() << " points\n";
        }
    }

    std::cout << "Single linkage clustering completed." << std::endl;

    // Output results
    std::cout << "Found " << clusters.size()
              << " clusters (min size = " << min_cluster_size << ")\n";
    for(size_t i = 0; i < clusters.size(); ++i){
        std::cout << "Cluster " << i << " (" 
                  << clusters[i].size() << " points): ";
        for(int p : clusters[i])
            std::cout << p << " ";
        std::cout << "\n";
    }

    return clusters;
}