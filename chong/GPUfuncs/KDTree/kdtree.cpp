#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <cmath>
#include <hip/hip_runtime.h>
#include <random>
#include <fstream>
#include <sstream>
#include <string>

using Point  = std::array<double,2>;
using PI     = std::pair<Point,int>;

struct KDNode {
    std::array<double,2> point;        // the 2D coordinates
    int                  axis;         // splitting axis: 0 or 1
    int                  index;        
    std::unique_ptr<KDNode> left, right; // pointer to left and right subtree

    KDNode(const std::array<double,2>& pt,
           int ax,
           int idx)
      : point(pt), axis(ax), index(idx),
        left(nullptr), right(nullptr)
    {}
};

/**
 * Read 2D points from a text or CSV file.
 *
 * Each non-empty line should contain exactly two numbers,
 * either separated by whitespace:
 *    1.23  4.56
 * or by comma:
 *    1.23,4.56
 *
 * Lines beginning with '#' or empty lines are skipped.
 */
std::vector<Point> readPointsFromFile(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::vector<Point> pts;
    std::string line;
    while (std::getline(in, line)) {
        // skip comments or blank lines
        if (line.empty() || line[0] == '#') 
            continue;

        // normalize commas to spaces
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream iss(line);
        double x, y;
        if (iss >> x >> y) {
            pts.push_back({x,y});
        } else {
            std::cerr << "Warning: skipping malformed line: " 
                      << line << "\n";
        }
    }
    return pts;
}


// Squared‐distance helper
inline double sqDist(const std::array<double,2>& a,
                     const std::array<double,2>& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return dx*dx + dy*dy;
}


// populates k-nn heap
void queryKNN(
    const KDNode*                                  node,
    const std::array<double,2>&                    query,
    int                                            query_idx,   
    int                                            k,
    std::priority_queue<std::pair<double,int>>&    heap,
    const std::vector<std::array<double,2>>&       points 
) {
    if (!node) return;

    // Skip the query point itself
    if (node->index == query_idx) {
        ;
    }
    else {
        double dist2 = sqDist(node->point, query);
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

    queryKNN(nearChild, query, query_idx, k, heap, points);
    
    // only recurse into far child if heap is not filled
    // or distance from query to splitting axis is smaller than current furthest neighbour -> potentially close points on other side of axis
    double slabDist2 = diff * diff;
    if (heap.size() < static_cast<size_t>(k) || slabDist2 < heap.top().first)
    {
        queryKNN(farChild, query, query_idx, k, heap, points);
    }
}

// recursive helper on a vector<PI>
std::unique_ptr<KDNode> buildKDTreeRecursive(
    std::vector<PI>&   arr,
    int                l, 
    int                r,
    int                depth
) {
    if (l >= r) return nullptr;
    int axis = depth % 2;
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

    // now arr[m] is the median pair
    auto [pt, idx] = arr[m];
    auto node = std::make_unique<KDNode>(pt, axis, idx);

    // build left and right subtree
    node->left  = buildKDTreeRecursive(arr, l,    m,     depth+1);
    node->right = buildKDTreeRecursive(arr, m+1,  r,     depth+1);
    return node;
}

// Facade Interface for easier use
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


// Replace tuple of each k-nn to every point with MRD instead of distance to point
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

// Sanity Check
void printAndVerifyCoreDists(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    int k
) {
    std::cout << "Point     CoreDist  Verified\n";
    std::cout << "-----------------------------\n";
    for (int i = 0; i < 10; ++i) {
        // Brute‐force: find the k-th nearest neighbor distance by scanning all others
        std::vector<double> dists;
        dists.reserve(points.size()-1);
        for (int j = 0; j < (int)points.size(); ++j) {
            if (i == j) continue;
            dists.push_back(std::sqrt(sqDist(points[i], points[j])));
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

// Sanity Check
void printAndVerifyMutualReachability(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    const std::vector<std::vector<std::pair<int,double>>>& knn_graph
) {
    std::cout << "\nVerifying mutual‐reachability distances:\n";
    std::cout << "i\tj\tcore_i\tcore_j\td_ij\tmreach_calc\tmreach_stored\tOK?\n";
    std::cout << "----------------------------------------------------------------\n";
    int N = points.size();
    for (int i = 0; i < 10; ++i) {
        for (auto const & pr : knn_graph[i]) {
            int j        = pr.first;
            double mr_st = pr.second;               // stored MR
            double d_ij  = std::sqrt(sqDist(points[i], points[j]));
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

int main(int argc, char** argv) {
    std::vector<Point> points;
    int k = NULL;
    // Parse param overrides (if any)
    int i = 1;
    while (i + 1 < argc) {
        if (!strcmp(argv[i], "--minpts")) {
            k = std::atoi(argv[i+1]);
            i += 2;
        }
        else if (!strcmp(argv[i], "--input")) {
            try{
                points = readPointsFromFile(argv[i+1]);
                i += 2;
                std::cout << "Read " << points.size() << " points.\n";
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                return 1;
            }
        }
        else {
            // unrecognized flag: skip just the flag
            std::cerr << "Warning: unknown option '" << argv[i] << "'\n";
            i += 1;
        }
    }
    if (k == NULL){
        k = 2;
    }

    int N = points.size();
    std::vector<std::vector<std::pair<int,double>>> knn_graph(points.size());
    std::vector<double> core_dist(points.size());

    auto root = buildKDTree(points);

    for (int i = 0; i < N; ++i) {
        // 1) Prepare an empty max-heap
        std::priority_queue<std::pair<double,int>> heap;

        // 2) Query the tree for point i
        queryKNN(root.get(), points[i], i, k, heap, points);

        // 3) Extract neighbors (and record the core distance)
        //    Since heap is max‐heap, after you pop k elements,
        //    the last popped distance = core distance
        double d_k = 0;
        // 1) Record core‐distance before you empty the heap
        double coreDistSq   = heap.top().first;
        core_dist[i]        = std::sqrt(coreDistSq);

        // 2) (Optionally) extract the k neighbors if you need the graph
        std::vector<std::pair<int,double>> nbrs;
        nbrs.reserve(k);
        while (!heap.empty()) {
        auto [d_sq, idx] = heap.top(); heap.pop();
        nbrs.emplace_back(idx, std::sqrt(d_sq));
        }
        // reverse to have them in ascending order if you like
        std::reverse(nbrs.begin(), nbrs.end());
        knn_graph[i] = std::move(nbrs);
    }
    printAndVerifyCoreDists(points, core_dist, k);

    // Calculate Mutual Reachability Distance
    convertToMutualReachability(knn_graph, core_dist);

    printAndVerifyMutualReachability(points, core_dist, knn_graph);
}
