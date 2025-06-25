#include "KdTreeBuilder.h"
#include "KdTypes.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <hip/hip_runtime.h>
#include <cmath>          
#include <stdexcept>

// C‐tor: allocate host buffers
KdTreeBuilder::KdTreeBuilder(int N, int D)
  : _N(N),
    _D(D),
      // levels = ceil(log2(N)) + 1
    _levels(int(std::ceil(std::log2(double(N)))) + 1),
  // nodeCount = 2^levels - 1
    _nodeCount((1ull << _levels) - 1),
    _levelSizes(_levels),
    _levelOffsets(_levels),
    h_coords(size_t(N) * size_t(D)),
    h_nodes(size_t(_nodeCount)),
    d_coords(nullptr),
    d_nodes(nullptr)
{
    for(int lvl = 0; lvl < _levels; ++lvl) {
        _levelSizes[lvl]   = std::min(1 << lvl, _N);
        _levelOffsets[lvl] = (lvl == 0
                            ? 0
                            : _levelOffsets[lvl-1] + _levelSizes[lvl-1]);
    }
    // just initialize your host nodes (tuple indices, null children)
    KdNode::initializeKdNodesArray(h_nodes.data(), _nodeCount);
}

// D‐tor: free device memory
KdTreeBuilder::~KdTreeBuilder() {
    if (d_coords) hipFree(d_coords);
    if (d_nodes)  hipFree(d_nodes);
    if(d_refArrays_dev)    hipFree(d_refArrays_dev);
    if(d_outRefArrays_dev) hipFree(d_outRefArrays_dev);
    if(d_levelSizes)   hipFree(d_levelSizes);
    if(d_levelOffsets) hipFree(d_levelOffsets);
    for(auto ptr : d_refArrays) {
        if(ptr) hipFree(ptr);
    }
    for(auto ptr : out_refArrays) {
        if(ptr) hipFree(ptr);
    }
}

// top‐level driver
void KdTreeBuilder::build() {
    allocateDeviceMemory();   // all hipMallocs + host→device copies
    initReferenceArrays();    // thrust::sequence or host→device to each d_refArrays[dim]
    sortReferenceArrays();    // your D sorts
    partitionLevels();        // Brown’s partition kernel loop
    retrieveResults();        // hipMemcpy device→host of h_nodes (or whatever)
}

//---------------------------------------------------------------------------
// Allocate GPU buffers for coords, refs and nodes, and copy coords over.
//---------------------------------------------------------------------------
void KdTreeBuilder::allocateDeviceMemory() {
    hipError_t err;

    // coords
    size_t coordBytes = sizeof(KdCoord) * size_t(_N) * size_t(_D);
    err = hipMalloc(&d_coords, coordBytes);
    if(err != hipSuccess) throw std::runtime_error("hipMalloc d_coords failed");

    // nodes
    size_t nodesBytes = sizeof(KdNode) * _nodeCount;
    err = hipMalloc(&d_nodes, nodesBytes);
    if (err != hipSuccess)
      throw std::runtime_error("hipMalloc d_nodes failed");

    // per-dimension refs
    d_refArrays.resize(_D);
    out_refArrays.resize(_D);
    for(int dim = 0; dim < _D; ++dim) {
      err = hipMalloc(&d_refArrays[dim],
                      sizeof(refIdx_t) * size_t(_N));
      if(err != hipSuccess)
        throw std::runtime_error("hipMalloc d_refArrays[" +
                                 std::to_string(dim) + "] failed");
      err = hipMalloc(&out_refArrays[dim],
                      sizeof(refIdx_t) * size_t(_N));
      if(err != hipSuccess)
        throw std::runtime_error("hipMalloc out_refArrays[" +
                                 std::to_string(dim) + "] failed");
    }

        // allocate device‐side pointer arrays
    hipMalloc(&d_refArrays_dev,    sizeof(refIdx_t*) * _D);
    hipMalloc(&d_outRefArrays_dev, sizeof(refIdx_t*) * _D);
    // copy host→device pointer lists
    hipMemcpy(d_refArrays_dev,
            d_refArrays.data(),
            sizeof(refIdx_t*) * _D,
            hipMemcpyHostToDevice);
    hipMemcpy(d_outRefArrays_dev,
            out_refArrays.data(),
            sizeof(refIdx_t*) * _D,
            hipMemcpyHostToDevice);


    // upload host→device: coords and initial nodes
    err = hipMemcpy(d_coords,
                    h_coords.data(),
                    coordBytes,
                    hipMemcpyHostToDevice);
    if(err != hipSuccess) throw std::runtime_error("hipMemcpy d_coords failed");

    err = hipMemcpy(d_nodes,
                    h_nodes.data(),
                    nodesBytes,
                    hipMemcpyHostToDevice);
    if(err != hipSuccess) throw std::runtime_error("hipMemcpy d_nodes failed");
    // 4) levelSizes / levelOffsets
    size_t lvlBytes = sizeof(int) * size_t(_levels);
    // allocate on device
    hipMalloc(&d_levelSizes,   lvlBytes) ;
    hipMalloc(&d_levelOffsets, lvlBytes) ;
    // copy host→device
    hipMemcpy(d_levelSizes,
                _levelSizes.data(),
                lvlBytes,
                hipMemcpyHostToDevice) ;
    hipMemcpy(d_levelOffsets,
                _levelOffsets.data(),
                lvlBytes,
                hipMemcpyHostToDevice) ;
}

// fill d_refs with 0,1,2,…,N-1 for each of the D dims
void KdTreeBuilder::initReferenceArrays() {
  for(int dim = 0; dim < _D; ++dim) {
    thrust::device_ptr<refIdx_t> devPtr(d_refArrays[dim]);
    thrust::sequence(devPtr, devPtr + _N);
    thrust::device_ptr<refIdx_t> devPtr2(out_refArrays[dim]);
    thrust::sequence(devPtr2, devPtr2 + _N);
  }
}

// 1) A small comparator that pulls out the right coordinate
struct RefCompare {
  const KdCoord* coords;  // flattened [N*D]
  int D, dim;

  __host__ __device__
  bool operator()(refIdx_t a, refIdx_t b) const {
    // compare coordinate-‘dim’ of tuple a vs tuple b
    return coords[a * D + dim] < coords[b * D + dim];
  }
};

void KdTreeBuilder::sortReferenceArrays() {
  // for each dimension, sort its own index array
  for(int dim = 0; dim < _D; ++dim) {
    // wrap the device pointer for Thrust
    thrust::device_ptr<refIdx_t> devIn(d_refArrays[dim]);

    // 1) seed it to [0,1,2,…,N-1]
    thrust::sequence(thrust::device, devIn, devIn + _N);

    // 2) sort by the `dim`th coordinate
    thrust::sort(
      thrust::device,
      devIn, 
      devIn + _N,
      RefCompare{ d_coords, _D, dim }
    );

    // 3) (optional) copy into your out-buffer so it's ready for lvl 0
    //    partitionLevels() will overwrite it at lvl 0, so you can skip this if you like.
    hipMemcpy(
      out_refArrays[dim],
      thrust::raw_pointer_cast(devIn),
      sizeof(refIdx_t) * _N,
      hipMemcpyDeviceToDevice
    );
  }
}


// store host coords for later
void KdTreeBuilder::setPoints(const std::vector<KdCoord>& coords) {
    if ((int)coords.size() != _N*_D)
        throw std::runtime_error("setPoints: size mismatch");
    h_coords = coords;
}


extern "C" __global__ void partition_level_kernel(
    KdNode*        nodes,          // [nodeCount]
    const KdCoord* coords,         // [N*D]
    refIdx_t**     inRefArrays,    // [D][N]
    refIdx_t**     outRefArrays,   // [D][N]
    int            N,              // # points
    int            D,              // dims
    int            level,          // ℓ = depth (0=root)
    int            segmentSize,    // ceil(N/2^ℓ)
    int            axis,           // ℓ % D
    int*     d_levelSizes,   // [levels]
    int*     d_levelOffsets,  // [levels]
    int      levels
) {
    // 1) flatten 2D launch → subtree in [0..levelSizes[level])
    int bx      = blockIdx.x;
    int by      = blockIdx.y;
    int gx      = gridDim.x;
    int subtree = by*gx + bx;
    int numSub  = d_levelSizes[level];
    if(subtree >= numSub) return;

    // 2) compute slice [start..end)
    int start = subtree * segmentSize;
    int end   = min(start + segmentSize, N);
    int count = end - start;
    if(count <= 0) return;

    // 3) pick median tuple‐ID and value
    int midPos    = start + (count/2);
    refIdx_t medID = inRefArrays[axis][midPos];
    KdCoord medVal = coords[medID*D + axis];

    // 4) compute where in nodes[] to write
    int baseIndex = d_levelOffsets[level] + subtree;
    nodes[baseIndex].tuple = medID;

    // 5) children only if we're *not* at the last level
    if(level + 1 < levels) {
      int nextSize   = d_levelSizes[level+1];
      int nextOffset = d_levelOffsets[level+1];
      int leftID     = subtree*2;
      int rightID    = leftID + 1;

      nodes[baseIndex].ltChild = 
         (leftID  < nextSize ? nextOffset + leftID  : -1);
      nodes[baseIndex].gtChild =
         (rightID < nextSize ? nextOffset + rightID : -1);
    } else {
      // leaf nodes have no children
      nodes[baseIndex].ltChild =
      nodes[baseIndex].gtChild = -1;
    }

    // 6) partition *all* D sorted index‐lists into outRefArrays
    if(threadIdx.x == 0) {
      for(int d = 0; d < D; ++d) {
        int writeL = start;
        int writeR = start + (count/2);
        for(int i = start; i < end; ++i) {
          refIdx_t rid = inRefArrays[d][i];
          KdCoord v    = coords[rid*D + axis];
          if(v < medVal)
            outRefArrays[d][ writeL++ ] = rid;
          else
            outRefArrays[d][ writeR++ ] = rid;
        }
      }
    }
}


void KdTreeBuilder::partitionLevels() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    const int maxGridX = prop.maxGridSize[0];
    const int threadsPerBlock = 256;
    refIdx_t** inDev  = d_refArrays_dev;
    refIdx_t** outDev = d_outRefArrays_dev;
    for(int lvl = 0; lvl < _levels; ++lvl) {
    int numSubtrees = 1 << lvl;               // 2^lvl subtrees at this depth
    int segmentSize    = (_N + numSubtrees - 1)/numSubtrees;
    int axis       = lvl % _D;                 // cycle axes: x, y, z, x, …

    // split numSub across X and Y so X<=maxGridX
    int gx = std::min(numSubtrees, maxGridX);
    int gy = (numSubtrees + gx - 1) / gx;
    dim3 grid(gx, gy), block(threadsPerBlock);
    hipLaunchKernelGGL(
        partition_level_kernel,
        grid,   // one block per subtree
        block,
        0, 0,
        d_nodes,
        d_coords,
        inDev,  // pointer-of-pointer to all D sorted lists
        outDev,
        _N, _D,
        lvl,
        segmentSize,
        axis,
        d_levelSizes,      // <-- new
        d_levelOffsets,    // <-- new
        _levels
    );
    hipDeviceSynchronize();
    // swap roles for next level:
    std::swap(inDev, outDev);
    }
}


// give main() access to the built nodes[]
const std::vector<KdNode>& KdTreeBuilder::getNodes() const {
    return h_nodes;
}


//---------------------------------------------------------------------------
// Copy back the built KD‐nodes from device to host vector.
//---------------------------------------------------------------------------
void KdTreeBuilder::retrieveResults() {
    size_t nodesBytes = sizeof(KdNode) * size_t(_N);
    hipError_t err = hipMemcpy(h_nodes.data(),
                               d_nodes,
                               nodesBytes,
                               hipMemcpyDeviceToHost);
    if (err != hipSuccess)
        throw std::runtime_error("hipMemcpy from d_nodes failed");
}
