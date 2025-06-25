#include "KdTreeBuilder.h"
#include "KdTypes.h"
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cassert>

// returns true if subtree rooted at `nodeIdx` is valid
bool validateKdTree(
    const std::vector<KdNode>& nodes,
    const std::vector<KdCoord>& coords,
    int        nodeIdx,
    int        D,
    std::vector<KdCoord>& minBounds,
    std::vector<KdCoord>& maxBounds,
    int        depth = 0
) {
    if (nodeIdx < 0) return true; 
    assert(nodeIdx < (int)nodes.size());
    
    const KdNode &n = nodes[nodeIdx];
    int axis = depth % D;
    int tuple = n.tuple;
    
    // check that point lies within [minBounds, maxBounds]
    for(int d = 0; d < D; ++d) {
      auto v = coords[tuple*D + d];
      if (v < minBounds[d] || v > maxBounds[d]) {
        std::cerr<<"Node "<<nodeIdx
                 <<" (tuple="<<tuple
                 <<") violates bounds on dim "<<d
                 <<" (value="<<v
                 <<", allowed=["<<minBounds[d]
                 <<","<<maxBounds[d]<<"])\n";
        return false;
      }
    }
    
    // save the split coordinate
    auto splitVal = coords[tuple*D + axis];
    
    // recurse left with updated max bound on this axis
    KdCoord oldMax = maxBounds[axis];
    maxBounds[axis] = splitVal;
    bool okL = validateKdTree(nodes, coords, n.ltChild,
                              D, minBounds, maxBounds, depth+1);
    maxBounds[axis] = oldMax;             // restore
    
    // recurse right with updated min bound on this axis
    KdCoord oldMin = minBounds[axis];
    minBounds[axis] = splitVal;
    bool okR = validateKdTree(nodes, coords, n.gtChild,
                              D, minBounds, maxBounds, depth+1);
    minBounds[axis] = oldMin;             // restore
    
    return okL && okR;
}

int main() {
    // N = number of points
    // D = number of dimensions
    const int N = 16, D = 2;
    // initialising dummy data
    std::vector<KdCoord> points(N*D, 0.0f);
    // fill something trivial
    for (int i = 0; i < N*D; ++i) points[i] = float(i);

    KdTreeBuilder B(N, D);
    // set h_coords == points
    B.setPoints(points);
    B.build();    // will call allocateDeviceMemory() → initRef → sort → partition (stub) → retrieve
    auto nodes = B.getNodes();
    auto const & coords = B.getPoints();   // you’ll need to expose this
    for(int i = 0; i < N; ++i) {
        std::cout << "Tree node " << i << " lives at tuple index "
                << nodes[i].tuple << " → point: (";
        int idx = nodes[i].tuple;
        for(int d = 0; d < D; ++d) {
            std::cout << coords[idx*D + d]
                    << (d+1< D ? ", " : "");
        }
        std::cout << ")\n";
    }
    // initialize infinite bounds
    std::vector<KdCoord> minB(D, std::numeric_limits<KdCoord>::lowest());
    std::vector<KdCoord> maxB(D, std::numeric_limits<KdCoord>::max());

    bool ok = validateKdTree(nodes, coords, 0 /*root idx*/, D, minB, maxB);
    std::cout<<"KD-Tree valid? "<<(ok?"YES":"NO")<<"\n";
    std::cout<<"First node tuple index: "<<nodes[0].tuple<<"\n";
    return 0;
}
