#pragma once
#include <cstdint>

using KdCoord = float;
using refIdx_t  = int32_t;

// A KD‐tree node: stores the tuple index, and indices of its two children.
// Based on the original KdNode.h interface.
struct KdNode {
    refIdx_t tuple;
    refIdx_t ltChild;
    refIdx_t gtChild;
    static void initializeKdNodesArray(KdNode out[], size_t N) {
        for(size_t i=0;i<N;++i){
        out[i].tuple    = refIdx_t(i);
        out[i].ltChild  = out[i].gtChild = -1;
        }
    }
};
