#include <thrust/sort.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <functional>

struct Edge { int u, v; float mrd; };

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
        collect_members(right_child[c],N_pts, left_child, right_child, out);
    }
}

/*
    Single Linkage Algorithm 
    Starts Each Point as a Singleton Cluster
    Based on Decreasing Lambda = (1/MRD) {Increasing MRD}, Merge Clusters

    parent[i] = Tracks Parent of Subclusters
    sz[i] = Tracks Number Of Points in Cluster i (Don't store all point indexes since merging clusters will take more time)
    birth_lambda[i] = Lambda when Cluster was formed
    death_lambda[i] = Lambda when Cluster merges to form a bigger cluster
    stability[i] = birth_lambda - death_lambda * sz[i]

*/
int main(){
    // assume `mst_edges` is your input vector<Edge> of size N–1
    std::sort(mst_edges.begin(), mst_edges.end(),
                [] (const Edge &a, const Edge &b) { return a.mrd < b.mrd; });

    // After sorting:
    assert(!mst_edges.empty());
    for (auto &e : mst_edges) {
        assert(e.u >= 0 && e.u < N_pts);
        assert(e.v >= 0 && e.v < N_pts);
        assert(e.mrd > 0);
    }
    
    // int N_pts = /* number of points */;
    int max_clusters = 2*N_pts;         // All p
    std::vector<int> parent(max_clusters), sz(max_clusters);
    std::vector<float> birth_lambda(max_clusters), death_lambda(max_clusters), stability(max_clusters);
    std::vector<int> left_child(max_clusters, -1), right_child(max_clusters, -1);
    
    // 2. Guard against zero‐length edges (just in case)
    float smallest_mrd = mst_edges.front().mrd;
    if (smallest_mrd <= 0.f) {
        // handle degenerate case (e.g. set to a tiny epsilon)
        smallest_mrd = std::numeric_limits<float>::min();
    }

    /* Compute lambda_max as the inverse of smallest_mrd. 
       Prevents Singleton Clusters from being most stable.
    */
    float lambda_max = 1.f / smallest_mrd;
    /* initialise all points as singleton clusters 
       singleton clusters have cluster ids within [0,N_pts-1]
    */
    for(int i = 0; i < N_pts; ++i){
        parent[i]       = i;
        sz[i]           = 1;
        birth_lambda[i]= lambda_max;
        death_lambda[i]= 0;
        stability[i]   = 0;
    }
    int next_cluster_id = N_pts;

    // lambda function for finding parent of a cluster
    // includes two pass path compression 
    // first pass to find root
    // second pass to update all nodes along the path to point to root
    auto find_root = [&](int x){
        // 1) Find the root
        int root = x;
        while(parent[root] != root)
            root = parent[root];
        // 2) Compress the path
        while(parent[x] != root){
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    // lambda function for finding root node of vertex in an edge 
    for(auto &e : mst_edges){
        int c1 = find_root(e.u),
            c2 = find_root(e.v);
        // if clusters are already connected, continue
        if(c1 == c2){
            continue;
        }

        // else make new cluster
        float lambda = 1.f / e.mrd;
        // record death of c1, c2
        death_lambda[c1] = lambda;
        death_lambda[c2] = lambda;
        // update their stability contributions
        stability[c1] += (birth_lambda[c1] - death_lambda[c1]) * sz[c1];
        stability[c2] += (birth_lambda[c2] - death_lambda[c2]) * sz[c2];

        // make new cluster
        int c_new = next_cluster_id++;
        parent[c1] = parent[c2] = c_new;
        parent[c_new] = c_new;
        sz[c_new]         = sz[c1] + sz[c2];
        birth_lambda[c_new] = lambda;
        stability[c_new]   = 0;
        death_lambda[c_new] = 0;
        left_child[c_new]  = c1;
        right_child[c_new] = c2;
    }

    // initialise all remaining singleton clusters to die at lambda = 0
    for(int c = 0; c < next_cluster_id; ++c){
        if(parent[c] == c){
            death_lambda[c] = 0;
            stability[c]   += (birth_lambda[c] - death_lambda[c]) * sz[c];
        }
    }

    // 1) Build a list of candidate cluster IDs:
    std::vector<int> candidates;
    for(int c = 0; c < next_cluster_id; ++c) {
        if (sz[c] >= min_cluster_size && death_lambda[c] > 0)  // <-- drop the root
            candidates.push_back(c);
    }


    // 2) Sort them by descending stability:
    std::sort(candidates.begin(), candidates.end(),
          [&](int a, int b){
              return stability[a] > stability[b];
          });

    
    // 3a) Pick the “most stable” clusters, skipping any whose direct children
    //     are already selected
    //     final clusters are built in descending stability as well
    std::vector<bool> is_selected(max_clusters, false);
    std::vector<int> final_clusters;
    for(int c : candidates) {
        int L = left_child[c], R = right_child[c];
        // if either child is itself a selected cluster, skip this one
        if ((L >= N_pts && is_selected[L]) ||
            (R >= N_pts && is_selected[R])) {
        continue;
        }
        // otherwise select it
        is_selected[c] = true;
        final_clusters.push_back(c);
    }
    // 3b) Now assign each point to the first (most-stable) selected cluster it appears in
    std::vector<int> assignment(N_pts, -1);
    std::vector<std::vector<int>> clusters;
    for(int c : final_clusters){
        // gather all member points under cluster c
        std::vector<int> mem;
        collect_members(c, N_pts, left_child, right_child, mem);

        // build this cluster’s final list of *newly* assigned points
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

    // 4) Output
    std::cout << "Found " << clusters.size()
            << " clusters (min size = " << min_cluster_size << ")\n";
    for(size_t i = 0; i < clusters.size(); ++i){
        std::cout << "Cluster " << i << " (" 
                << clusters[i].size() << " points): ";
        for(int p : clusters[i])
            std::cout << p << " ";
        std::cout << "\n";
    }
}
