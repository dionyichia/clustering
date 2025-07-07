#include "kd_tree/include/kdtree.hpp"
#include <algorithm>
#include <utility>
#include <cmath>        
#include <iostream>    
#include <queue>
#include <vector>

// KDNode constructor
KDNode::KDNode(Point pt, int ax, int idx)
  : point(std::move(pt)), axis(ax), index(idx) {}

  std::unique_ptr<KDNode> buildKDTreeRecursive(
    std::vector<PI>&   arr, // array of vectors stored as ((vector), index)
    int                l, 
    int                r,
    int                depth
) {
    if (l >= r) return nullptr;
    int D = int(arr[0].first.size());
    int axis = depth % D;
    int m    = (l + r) / 2;

    // ensure median index-th element is in correct place (if list were to be fully sorted)
    // elements before median index-th element are not sorted but are all < median index-th element
    // elements after median index-th element are not sorted but are all > median index-th element
    std::nth_element(
        arr.begin() + l,
        arr.begin() + m,
        arr.begin() + r,
        [axis](auto &a, auto &b) {
            return a.first[axis] < b.first[axis];
        }
    );

    // arr[m] is median vector of ( (point), index)
    auto node = std::make_unique<KDNode>(arr[m].first, axis, arr[m].second);

    // build left and right subtree
    node->left  = buildKDTreeRecursive(arr, l,    m,     depth+1);
    node->right = buildKDTreeRecursive(arr, m+1,  r,     depth+1);
    return node;
}

std::unique_ptr<KDNode> buildKDTree(std::vector<Point> pts) {
    // 1) make a vector of (point, original_index)
    std::vector<PI> work;
    work.reserve(pts.size());
    // allocate index to points arbitrarily 
    for (int i = 0; i < (int)pts.size(); ++i)
        work.emplace_back(pts[i], i);

    // 2) call the recursive helper on the PI-vector
    // returns root of KDTree
    return buildKDTreeRecursive(work, 0, work.size(), 0);
}

void queryKNN(
    const KDNode*                                  node,
    const Point&                                   query,
    int                                            query_idx,   
    int                                            k,
    std::priority_queue<std::pair<double,int>>&    heap,
    const std::vector<Point>&                      points,
    DistanceMetric                                metric, 
    float                                           p
) {
    if (!node) return;

    // Skip the query point itself
    if (node->index == query_idx) {
        ;
    }
    else {
        double dist2;
        dist2 = distance(query,node->point,metric,p);
        // if heap is not full, add point to heap containing k-nn to point
        if (heap.size() < static_cast<size_t>(k)) {
            heap.emplace(dist2, node->index);
        // otherwise only add if it is closer than current furthest point, heapify afterward
        } else if (dist2 < heap.top().first) {
            heap.pop();
            heap.emplace(dist2, node->index);
        }
    }

    int axis = node->axis;
    double diff = query[axis] - node->point[axis];
    // if diff < 0, query point is below splitting axis, so recurse into left sub child 
    const KDNode* nearChild = diff < 0 ? node->left.get() : node->right.get();
    const KDNode* farChild  = diff < 0 ? node->right.get(): node->left.get();

    queryKNN(nearChild, query, query_idx, k, heap, points,metric,p);
    
    // only recurse into far child if heap is not filled
    // or distance from query to splitting axis is smaller than current furthest neighbour -> potentially close points on other side of axis
    double diffAbs = std::abs(diff);
    bool shouldVisitFar = false;
    if (metric == DistanceMetric::EuclideanSquared) {
    shouldVisitFar = (heap.size() < k || diff*diff < heap.top().first);
    }
    else if (metric == DistanceMetric::Manhattan) {
    // the minimum extra cost to cross the splitting plane is |diff|
    shouldVisitFar = (heap.size() < k || diffAbs < heap.top().first);
    }
    else if (metric == DistanceMetric::Chebyshev) {
    // crossing the plane you incur at least |diff| in the ∞-norm
    shouldVisitFar = (heap.size() < k || diffAbs < heap.top().first);
    }
    else if (metric == DistanceMetric::Minkowski) {
    // crossing the plane adds at least |diff|^p to the p-sum,
    // so compare diffAbs^p vs (heap.top())^p or pre-compare after raising both to p.
    double worstP = std::pow(heap.top().first, p);
    shouldVisitFar = (heap.size() < k || std::pow(diffAbs, p) < worstP);
    }
    else if (metric == DistanceMetric::DSO) {
    if (query[axis] != 0.0) {
      double rel = diff / query[axis];
      shouldVisitFar = (heap.size()<k) || (rel*rel < heap.top().first);
    } else {
      // can’t prune on this axis if query[axis] is zero
      shouldVisitFar = true;
    }
    }
    if (shouldVisitFar) {
    queryKNN(farChild, query, query_idx, k, heap, points,metric,p);
    }
}

void convertToMutualReachability(
    std::vector<std::vector<std::pair<int,double>>>& knn_graph,
    const std::vector<double>& core_dist
) {
    int N = knn_graph.size();
    // for all points
    for (int i = 0; i < N; ++i) {
        // range-for operation which iterates over all k nearest neighbours
        for (auto& pr : knn_graph[i]) {
            int j      = pr.first;   // neighbor’s index
            double dij = pr.second;  // original d(i,j)
            // mutual‐reachability distance
            pr.second = std::max({ core_dist[i],
                                   core_dist[j],
                                   dij });
        }
    }
}

void printAndVerifyCoreDists(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    int k,
    DistanceMetric metric,
    float p
) {
    std::cout << "Point     CoreDist  Verified\n";
    std::cout << "-----------------------------\n";
    for (int i = 0; i < 10; ++i) {
        // Brute‐force: find the k-th nearest neighbor distance by scanning all others
        std::vector<double> dists;
        dists.reserve(points.size()-1);
        for (int j = 0; j < (int)points.size(); ++j) {
            if (i == j) continue;
            if (metric == DistanceMetric::EuclideanSquared || metric == DistanceMetric::DSO){
                dists.push_back(std::sqrt(distance(points[i], points[j],metric,p)));
            }
            else{
                dists.push_back(distance(points[i], points[j],metric,p));
            }
        }
        std::nth_element(dists.begin(), dists.begin() + (k-1), dists.end());
        double brute_kth = dists[k-1];

        bool ok = std::abs(brute_kth - core_dist[i]) < 1e-6;
        std::cout
            << "(" << points[i][0] << "," << points[i][1] << ")   "
            << core_dist[i] << "    ";

        if (ok) {
            std::cout << "correct";
        } else {
            std::cout << "wrong (got " << brute_kth << ")";
        }
        std::cout << "\n";

    }
}
void printAndVerifyMutualReachability(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    const std::vector<std::vector<std::pair<int,double>>>& knn_graph,
    DistanceMetric metric,
    float p
) {
    std::cout << "\nVerifying mutual‐reachability distances:\n";
    std::cout << "i\tj\tcore_i\tcore_j\td_ij\tmreach_calc\tmreach_stored\tOK?\n";
    std::cout << "----------------------------------------------------------------\n";
    int N = points.size();
    for (int i = 0; i < 10; ++i) {
        for (auto const & pr : knn_graph[i]) {
            int j        = pr.first;
            double mr_st = pr.second;               // stored MR
            double d_ij;
            if(metric == DistanceMetric::EuclideanSquared || metric == DistanceMetric::DSO){
                d_ij = std::sqrt(distance(points[i], points[j],metric,p));
            }
            else{
                d_ij = distance(points[i], points[j],metric,p);
            }
            double mr_cf = std::max({ core_dist[i],
                                      core_dist[j],
                                      d_ij });     // recalculated MR
            bool ok      = std::abs(mr_cf - mr_st) < 1e-6;
            std::cout 
                << i << '\t'
                << j << '\t'
                << core_dist[i] << '\t'
                << core_dist[j] << '\t'
                << d_ij << '\t'
                << mr_cf << '\t'
                << mr_st << '\t'
                << (ok ? "correct" : "wrong")
                << "\n";
        }
    }
}

std::vector<Edge> flatten(const std::vector<std::vector<std::pair<int,double>>>& knn_graph) {
    std::vector<Edge> edges;
    
    // Calculate total number of edges (k * N)
    size_t total_edges = 0;
    for (const auto& neighbors : knn_graph) {
        total_edges += neighbors.size();
    }
    edges.reserve(total_edges);
    
    // Iterate through each point and its neighbors
    for (uint i = 0; i < knn_graph.size(); ++i) {
        for (const auto& neighbor : knn_graph[i]) {
            uint neighbor_idx = static_cast<uint>(neighbor.first);
            float mrd = static_cast<float>(neighbor.second);
            
            // Create edge from point i to its neighbor
            edges.emplace_back(i, neighbor_idx, mrd);
        }
    }
    
    return edges;
}

void printFirstNEdges(const std::vector<Edge>& edges, int n) {
    std::cout << "\n=== First " << n << " Edges ===\n";
    std::cout << "Format: (u -> v, weight)\n";
    std::cout << "------------------------\n";
    
    int limit = std::min(n, static_cast<int>(edges.size()));
    for (int i = 0; i < limit; ++i) {
        const Edge& e = edges[i];
        std::cout << "Edge " << i << ": (" 
                  << e.u << " -> " << e.v << ", " 
                  << e.weight << ")\n";
    }
    std::cout << "========================\n\n";
}
