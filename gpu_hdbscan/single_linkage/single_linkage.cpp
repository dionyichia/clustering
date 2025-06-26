#include "single_linkage.hpp"
#include "boruvka/boruvka.hpp"  // Include for Edge definition
#include <algorithm>
#include <limits>
#include <cassert>

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
    // Make a copy of mst_edges for sorting
    std::vector<Edge> edges_copy = mst_edges;
    
    // Sort edges by weight
    std::sort(edges_copy.begin(), edges_copy.end(),
              [](const Edge &a, const Edge &b) { return a.weight < b.weight; });

    // Validation
    assert(!edges_copy.empty());
    for (auto &e : edges_copy) {
        assert(e.u >= 0 && e.u < N_pts);
        assert(e.v >= 0 && e.v < N_pts);
        assert(e.weight > 0);
    }
    
    int max_clusters = 2 * N_pts;
    std::vector<int> parent(max_clusters), sz(max_clusters);
    std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // Guard against zero-length edges
    float smallest_weight = edges_copy.front().weight;
    if (smallest_weight <= 0.f) {
        smallest_weight = std::numeric_limits<float>::min();
    }

    // Compute lambda_max as the inverse of smallest_mrd
    float lambda_max = 1.f / smallest_weight;
    
    // Initialize all points as singleton clusters
    for(int i = 0; i < N_pts; ++i){
        parent[i] = i;
        sz[i] = 1;
        birth_lambda[i] = lambda_max;
        death_lambda[i] = 0;
        stability[i] = 0;
    }
    int next_cluster_id = N_pts;

    // Union-find with path compression
    auto find_root = [&](int x){
        // Find the root
        int root = x;
        while(parent[root] != root)
            root = parent[root];
        // Compress the path
        while(parent[x] != root){
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    // Process edges to build hierarchy
    for(auto &e : edges_copy){
        int c1 = find_root(e.u),
            c2 = find_root(e.v);
        
        if(c1 == c2){
            continue;
        }

        float lambda = 1.f / e.weight;
        // Record death of c1, c2
        death_lambda[c1] = lambda;
        death_lambda[c2] = lambda;
        // Update stability contributions
        stability[c1] += (birth_lambda[c1] - death_lambda[c1]) * sz[c1];
        stability[c2] += (birth_lambda[c2] - death_lambda[c2]) * sz[c2];

        // Create new cluster
        int c_new = next_cluster_id++;
        parent[c1] = parent[c2] = c_new;
        parent[c_new] = c_new;
        sz[c_new] = sz[c1] + sz[c2];
        birth_lambda[c_new] = lambda;
        stability[c_new] = 0;
        death_lambda[c_new] = 0;
        left_child[c_new] = c1;
        right_child[c_new] = c2;
    }

    // Initialize remaining singleton clusters to die at lambda = 0
    for(int c = 0; c < next_cluster_id; ++c){
        if(parent[c] == c){
            death_lambda[c] = 0;
            stability[c] += (birth_lambda[c] - death_lambda[c]) * sz[c];
        }
    }

    // Build candidate cluster list
    std::vector<int> candidates;
    for(int c = 0; c < next_cluster_id; ++c) {
        if (sz[c] >= min_cluster_size && death_lambda[c] > 0)
            candidates.push_back(c);
    }

    // Sort by descending stability
    std::sort(candidates.begin(), candidates.end(),
              [&](int a, int b){
                  return stability[a] > stability[b];
              });

    // Select most stable clusters
    std::vector<bool> is_selected(max_clusters, false);
    std::vector<int> final_clusters;
    for(int c : candidates) {
        int L = left_child[c], R = right_child[c];
        if ((L >= N_pts && is_selected[L]) ||
            (R >= N_pts && is_selected[R])) {
            continue;
        }
        is_selected[c] = true;
        final_clusters.push_back(c);
    }

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
        if(!this_cluster.empty())
            clusters.push_back(std::move(this_cluster));
    }

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