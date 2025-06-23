#include <algorithm>
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
#include <cassert>
#include <limits>

using Point = std::vector<double>;
using PI     = std::pair<Point,int>;

struct KDNode {
  Point             point;   // was array<double,2>, now vector
  int               axis;    // will be set in [0…D−1]
  int               index;
  std::unique_ptr<KDNode> left, right;

  KDNode(Point pt, int ax, int idx)
    : point(std::move(pt)), axis(ax), index(idx) {}
};


//------------------------------------------------------------------------------
// Scale all points so that x and y each span [0,1]. 
// If all x's (or y's) are equal, they get mapped to 0.0.
//------------------------------------------------------------------------------
void normalizePoints(std::vector<Point>& pts) {
  if (pts.empty()) return;
  size_t D = pts[0].size();
  
  // 1) init mins/maxs
  std::vector<double> minV(D,  std::numeric_limits<double>::infinity());
  std::vector<double> maxV(D, -std::numeric_limits<double>::infinity());

  // 2) scan all points
  for (auto& p : pts) {
    for (size_t i = 0; i < D; ++i) {
      minV[i] = std::min(minV[i], p[i]);
      maxV[i] = std::max(maxV[i], p[i]);
    }
  }

  // 3) rescale
  for (auto& p : pts) {
    for (size_t i = 0; i < D; ++i) {
      double range = maxV[i] - minV[i];
      p[i] = (range > 0.0 ? (p[i] - minV[i]) / range : 0.0);
    }
  }
}


/**
 * Read points from a text or CSV file.
 *
 * Each non-empty line should contain features,
 * either separated by whitespace:
 *    1.23  4.56
 * or by comma:
 *    1.23,4.56
 *
 * Lines beginning with '#' or empty lines are skipped.
 */
std::vector<Point> readPointsFromFile(const std::string& filename, int dimensions) {
  std::ifstream in(filename);
  if (!in) throw std::runtime_error("Unable to open file");

  std::vector<Point> pts;
  std::string line;

  // Lines beginning with '#' or empty lines are skipped.
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);

    Point p;
    double val;
    while (iss >> val) {
      p.push_back(val);
    }
    if (p.size() != dimensions) {
      std::cerr << "Warning: skipping line with wrong dimension\n";
      continue;
    }
    pts.push_back(std::move(p));
  }
  return pts;
}


//-------------------------------------------------------------------------
// Metric selector
//-------------------------------------------------------------------------
enum class DistanceMetric {
    EuclideanSquared,  // ∑ (Δi)²
    Manhattan,         // ∑ |Δi|
    Chebyshev,         // max |Δi|
    Minkowski           // (∑ |Δi|^p)^(1/p)
};

// D-dimensional Minkowski
// if metric==Minkowski, you must supply a positive “p” (e.g. 3, 4, even non-integer).
inline double distance(
    const Point& a,
    const Point& b,
    DistanceMetric metric,
    float             p = 2.0
) {
    assert(a.size() == b.size());
    const size_t D = a.size();

    switch (metric) {
      case DistanceMetric::EuclideanSquared: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = a[i] - b[i];
          sum += d*d;
        }
        return sum;
      }

      case DistanceMetric::Manhattan: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          sum += std::abs(a[i] - b[i]);
        }
        return sum;
      }

      case DistanceMetric::Chebyshev: {
        double mx = 0.0;
        for (size_t i = 0; i < D; ++i) {
          mx = std::max(mx, std::abs(a[i] - b[i]));
        }
        return mx;
      }

      case DistanceMetric::Minkowski: {
        if (p <= 0.0) 
          throw std::invalid_argument("p must be > 0 for Minkowski");
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          sum += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(sum, 1.0 / p);
      }

      default:
        throw std::invalid_argument("Unknown DistanceMetric");
    }
}

// helper to turn enum into a human-readable name
static const char* metricName(DistanceMetric m) {
    switch (m) {
      case DistanceMetric::EuclideanSquared: return "EuclideanSquared";
      case DistanceMetric::Manhattan:        return "Manhattan";
      case DistanceMetric::Chebyshev:        return "Chebyshev";
      case DistanceMetric::Minkowski:         return "Minkowski";
    }
    return "Unknown";
}


// populates k-nn heap
void queryKNN(
    const KDNode*                                  node,
    const Point&                                   query,
    int                                            query_idx,   
    int                                            k,
    std::priority_queue<std::pair<double,int>>&    heap,
    const std::vector<Point>&                      points,
    DistanceMetric                                metric, 
    float                                           p = 2.0
) {
    if (!node) return;

    // Skip the query point itself
    if (node->index == query_idx) {
        ;
    }
    else {
        double dist2;
        dist2 = distance(node->point, query,metric,p);
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
    if (shouldVisitFar) {
    queryKNN(farChild, query, query_idx, k, heap, points,metric,p);
    }
}

// recursive helper on a vector<PI>
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

// Print CLI Usage Instructions
void printUsage(char* prog) {
  std::cerr << R"(Usage:
    )" << prog << R"( --dimensions D --minpts K --input [FILENAME] --distMetric M [--minkowskiP p]

Parameters:
  --dimensions D   Number of features per data point (integer > 0)
  --minpts K       Minimum points for core‐distance (integer > 0)
  --input FILE     Path to input file
  --distMetric M   Distance metric (1= Manhattan,2= Euclidean,3 = Chebyshev,4 = Minkowski)
  --minkowskiP p   P-Value for Minkowski Distance
)";
}

// Sanity Check
void printAndVerifyCoreDists(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    int k,
    DistanceMetric metric,
    float p = 2.0
) {
    std::cout << "Point     CoreDist  Verified\n";
    std::cout << "-----------------------------\n";
    for (int i = 0; i < 10; ++i) {
        // Brute‐force: find the k-th nearest neighbor distance by scanning all others
        std::vector<double> dists;
        dists.reserve(points.size()-1);
        for (int j = 0; j < (int)points.size(); ++j) {
            if (i == j) continue;
            if (metric == DistanceMetric::EuclideanSquared){
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

// Sanity Check
void printAndVerifyMutualReachability(
    const std::vector<Point>& points,
    const std::vector<double>& core_dist,
    const std::vector<std::vector<std::pair<int,double>>>& knn_graph,
    DistanceMetric metric,
    float p = 2.0
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
            if(metric == DistanceMetric::EuclideanSquared){
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

int main(int argc, char** argv) {
    std::vector<Point> points;
    int dimensions = NULL;
    int k = NULL;
    int metricChoice = NULL;
    float minkowskiP = NULL;
    DistanceMetric metric;

    /* Param Overrides 
        --dimensions : number of dimensions (features) of each data point
        --minpts (int): used in calculating core-distance and MRD. Doubles as --minclustersize for cluster extraction
        --input (string): name of file storing input data
        --distanceMetric (int): Choose Distance Metric [1: Manhattan, 2: Euclidean, 3: Chebyshev, 4:Minkowski]
        --minkowskiP (float): P-value for minkowski
    */
    int i = 1;
    while (i + 1 < argc) {
        if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
            printUsage(argv[0]);
            return 1;
        }
        else if(!strcmp(argv[i], "--dimensions")){
            try{
                dimensions = std::stoi(argv[i+1]);
                i += 2;
                std::cout << "Number Of Dimensions " << dimensions << "\n";
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--minpts")) {
            try{
                k = std::stoi(argv[i+1]);
                i += 2;
                std::cout << "Minimum Points " << k << "\n";
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--input")) {
            try{
                points = readPointsFromFile(argv[i+1],dimensions);
                normalizePoints(points);
                i += 2;
                std::cout << "Read " << points.size() << " points.\n";
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--distMetric")){
            try{
                metricChoice = std::stoi(argv[i+1]);
                switch (metricChoice){
                    case(1):
                        metric = DistanceMetric::Manhattan;
                        break;
                    case(2):
                        metric = DistanceMetric::EuclideanSquared;
                        break;
                    case(3):
                        metric = DistanceMetric::Chebyshev;
                        break;
                    case(4):
                        metric = DistanceMetric::Minkowski;
                        break;
                }
                std::cout << "Distance Metric Selected: " << metricName(metric) << "\n";
                i += 2; 
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--minkowskiP")){
            try{
                minkowskiP = std::stof(argv[i+1]);
                std::cout << "Minkowski P Value: " << minkowskiP << "\n";
                i += 2;
            } catch(const std::exception& e) {
                std::cerr << e.what() << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else {
            // unrecognized flag: skip just the flag
            std::cerr << "Warning: unknown option '" << argv[i] << "\n";
            printUsage(argv[0]);
            i += 1;
        }
    }
    if (k == NULL){
        k = 2;
    }
    if (dimensions == NULL){
        std::cerr << "Dimensions Of Data Not Provided" << "\n";
        printUsage(argv[0]);
        return 1;
    }
    if (metric == DistanceMetric::Minkowski && minkowskiP == NULL){
        std::cerr << "P-Value not provided" << "\n";
        printUsage(argv[0]);
        return 1;
    }

    int N = points.size();
    std::vector<std::vector<std::pair<int,double>>> knn_graph(points.size());
    std::vector<double> core_dist(points.size());

    auto root = buildKDTree(points);

    for (int i = 0; i < N; ++i) {
        // 1) Prepare an empty max-heap
        std::priority_queue<std::pair<double,int>> heap;

        // 2) Query the tree for point i
        queryKNN(root.get(), points[i], i, k, heap, points,metric,minkowskiP);
        // 3) Extract neighbors (and record the core distance)
        //    Since heap is max‐heap, after you pop k elements,
        //    the last popped distance = core distance
        double d_k = 0;
        // 1) Record core‐distance before you empty the heap
        double coreDist = heap.top().first;
        if(metric == DistanceMetric::EuclideanSquared){
            core_dist[i] = std::sqrt(coreDist);
        }
        else{
            core_dist[i] = coreDist;
        }
        // 2) (Optionally) extract the k neighbors if you need the graph
        std::vector<std::pair<int,double>> nbrs;
        nbrs.reserve(k);
        while (!heap.empty()) {
        auto [d_sq, idx] = heap.top(); heap.pop();
        if(metric == DistanceMetric::EuclideanSquared){
            nbrs.emplace_back(idx, std::sqrt(d_sq));
        }
        else{
            nbrs.emplace_back(idx, d_sq);
        }
        }
        // reverse to have them in ascending order if you like
        std::reverse(nbrs.begin(), nbrs.end());
        knn_graph[i] = std::move(nbrs);
    }
    printAndVerifyCoreDists(points, core_dist, k,metric,minkowskiP);

    // Calculate Mutual Reachability Distance
    convertToMutualReachability(knn_graph, core_dist);
    printAndVerifyMutualReachability(points, core_dist, knn_graph,metric,minkowskiP);
}
