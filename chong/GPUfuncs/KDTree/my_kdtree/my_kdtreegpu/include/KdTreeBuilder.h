#pragma once
#include "KdTypes.h"
#include <vector>

// Kernel prototype (put this in a header shared with your .hip.cpp)
extern "C" __global__ void partition_level_kernel(
    KdNode*       nodes,        // device KdNode array
    const KdCoord* coords,      // flattened N×D coords
    refIdx_t**     inRefArrays,    // [D] pointers, each to [N] sorted index lists
    refIdx_t**     outRefArrays, // [D] pointers to [N]
    int           N,            // number of points
    int           D,            // dimensionality
    int           level,        // current tree depth
    int           segmentSize,  // approx N / (2^level)
    int           axis,          // cut‐axis = level % D
    int* d_levelSizes,      // <-- new
    int* d_levelOffsets,     // <-- new
    int _levels
);
// A simple class to build the KD tree on GPU
// Stubs out the steps from the Russell Brown algorithm:
//  1) init D reference arrays
//  2) sort each by the p-th coordinate
//  3) breadth-first partition levels
//  4) assemble nodes[]
class KdTreeBuilder {
public:
    KdTreeBuilder(int N, int D);
    ~KdTreeBuilder();

    // supply your host‐side coords as a flat vector of size N*D
    void setPoints(const std::vector<KdCoord>& coords);

    // runs the entire build on the GPU
    void build();

    // retrieve the resulting tree
    const std::vector<KdNode>& getNodes() const;
    // retrieve coordinate list to match KDNodes
    const std::vector<KdCoord>& getPoints() const { return h_coords; }

private:
    int _N, _D;
    int    _levels;          // how many passes = ceil(log2(N))+1
    size_t _nodeCount;       // how many nodes = (1<<levels)-1

    std::vector<KdCoord> h_coords;
    std::vector<KdNode>  h_nodes;
    // these two hold, per-level:
    //   levelSizes[ℓ]   = how many splits at level ℓ  = min(2^ℓ, N)
    //   levelOffsets[ℓ] = where that level starts in the flat node array
    std::vector<int>    _levelSizes;
    std::vector<int>    _levelOffsets;
    // if need to debug if sorting the reference arrays is working, uncomment h_refArrays
    // std::vector<std::vector<refIdx_t>> h_refArrays;
    // device pointers
    KdCoord*  d_coords;
    KdNode*   d_nodes;
    std::vector<refIdx_t*> d_refArrays;    // store reference arrays
    std::vector<refIdx_t*> out_refArrays;    // store reference arrays
    refIdx_t** d_refArrays_dev   = nullptr;  // device copy of your host d_refArrays[]
    refIdx_t** d_outRefArrays_dev= nullptr;
    int* d_levelSizes    = nullptr;
    int* d_levelOffsets  = nullptr;

    // steps
    void allocateDeviceMemory(); // Allocate GPU buffers for coords, refs and nodes, and copy coords over.
    void initReferenceArrays();
    void sortReferenceArrays();
    void partitionLevels();
    void retrieveResults();

    // disable copying
    KdTreeBuilder(const KdTreeBuilder&) = delete;
    KdTreeBuilder& operator=(const KdTreeBuilder&) = delete;
};
