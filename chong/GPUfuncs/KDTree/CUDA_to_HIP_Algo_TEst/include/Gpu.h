//
//  Gpu.h
//  KdTreeGPUsms
//
//  Created by John Robinson on 7/15/15.
//  Copyright (c) 2015 John Robinson. All rights reserved.
/*
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSEARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef gpu_Gpu_h
#define gpu_Gpu_h

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


using namespace std;

#include "KdNode.h"
#include "HipErrorCheck.h"
class Gpu {
	// Gpu class constants;
	static const uint MAX_THREADS = 1024;
	static const uint MAX_BLOCKS = 1024;
	static const uint MAX_GPUS = 2;

	// Static variables to keep track of multiple GPUs and some setters.
private:
	static sint numGPUs;
	static Gpu* gpus[MAX_GPUS];
	static refIdx_t firstNode; // Used to pass the index of the first node between methods
	static KdNode gpu1stNode;	// Used to store root indices fur multiple GPUs

	static inline sint setNumGPUs(sint ng) {numGPUs = min(ng,MAX_GPUS); return numGPUs;}
	static inline Gpu* getGPU(int n) {return (n>numGPUs || n<0) ? NULL : gpus[n];}
	static inline void sync() {
		for (int gpuCnt = 0; gpuCnt < numGPUs; gpuCnt++) gpus[gpuCnt]->syncGPU();
	}

public:
	// These are the API methods used outside the class.  They hide any details about the GPUs from the main program.
	static sint     getNumGPUs() {return numGPUs;}
	static void     gpuSetup(int gpu_max, int threads, int blocks, int dim); // GPU discovery and multiple GPU static variable setup.
	static void     initializeKdNodesArray(KdCoord coordinates[], const sint numTuples, const sint dim);
	static void     mergeSort(sint end[], const sint numTuples, const sint dim);
	static refIdx_t buildKdTree(KdNode kdNodes[], const sint numTuples, const sint dim);
	static sint     verifyKdTree(KdNode kdNodes[], const sint root, const sint dim, const sint numTuples);
	static void     getKdTreeResults(KdNode kdNodes[], KdCoord coord[], const sint numTuples, const sint dim);
	static int      getNumThreads() {
		if (numGPUs==2) {
			if  (gpus[0] == NULL || gpus[1] == NULL) return 0;
			return (min(gpus[0]->numThreads,gpus[1]->numThreads));
		} else {
			if  (gpus[0] == NULL ) return 0;
			return (gpus[0]->numThreads);
		}
	}
	static int getNumBlocks() {
		if (numGPUs==2) {
			if  (gpus[0] == NULL || gpus[1] == NULL) return 0;
			return (min(gpus[0]->numBlocks,gpus[1]->numBlocks));
		} else {
			if  (gpus[0] == NULL ) return 0;
			return (gpus[0]->numBlocks);
		}
	}
	int getDeviceId() const { return devID; }
	// Device specific variables
private:
	sint 		numThreads; 	// Constant value holding the number of threads
	sint 		numBlocks; 		// Constant value holding the number of blocks
	sint 		devID; 			// The GPU device we are talking to.
	refIdx_t** 	d_references;	// Pointer to array of pointers to reference arrays
	KdCoord**  	d_values;		// Pointer to array of pointers to value arrays
	KdCoord* 	d_coord;		// Pointer to coordinate array
	KdNode* 	d_kdNodes;		// Pointer to array of KdNodes
	sint* 		d_end;         // Pointer to array of end values in th GPU
	hipStream_t stream;       // hip stream for this GPU
	hipEvent_t syncEvent;   	// hip sync events
	hipEvent_t start, stop;   // hip Timer events
	uint 		dimen;	       // Number of dimensions
	uint 		num;           // Number of tuples or points
	refIdx_t    rootNode;      // Store the root node here so all partitionDim rouotines can get to it.

private:
	// Constructor
	Gpu(int threads, int blocks, int dev, int dim) {
		devID = dev;
		numThreads = threads;
		if (numThreads>MAX_THREADS) numThreads = 0;
		numBlocks = blocks;
		if (numBlocks>MAX_BLOCKS) numBlocks = 0;
		d_references = new refIdx_t*[dim+2];
		for (sint i = 0;  i < dim + 2; i++) d_references[i] = NULL;
		d_values = new KdCoord*[dim+2];
		for (sint i = 0;  i < dim + 2; i++) d_values[i] = NULL;
		d_coord = NULL;
		d_kdNodes = NULL;
		d_end = NULL;
		d_mpi = NULL;
		d_segLengthsLT = NULL;
		d_segLengthsGT = NULL;
		d_midRefs[0] = NULL;
		d_midRefs[1] = NULL;

		setDevice();
		HIP_CHECK(hipStreamCreate(&stream));
		HIP_CHECK(hipEventCreate(&syncEvent));
		HIP_CHECK(hipEventCreate(&start));
		HIP_CHECK(hipEventCreate(&stop));
		dimen = dim;
		num = 0;
	}


	~Gpu(){
		setDevice();
		if (d_references != NULL) {
			for (sint i = 0; i < dimen+1; i++)
				HIP_CHECK(hipFree(d_references[i]));
		}
		free(d_references);
		if (d_values != NULL) {
			for (sint i = 0; i < dimen+1; i++)
				HIP_CHECK(hipFree(d_values[i]));
		}
		free(d_values);
		if (d_coord != NULL)
			HIP_CHECK(hipFree(d_coord));
		if (d_coord != NULL)
			HIP_CHECK(hipFree(d_kdNodes));
		if (d_end != NULL)
			HIP_CHECK(hipFree(d_end));
		HIP_CHECK(hipEventDestroy(start));
		HIP_CHECK(hipEventDestroy(stop));
		HIP_CHECK(hipStreamDestroy(stream));
		num = 0;
	}

private:
	// These are the per GPU methods.  They implemented in Gpu.cu
	void inline setDevice() {HIP_CHECK(hipSetDevice(devID));}
	void initializeKdNodesArrayGPU(const KdCoord coord[], const int numTuples, const int dim);
	void initializeReferenceGPU(const sint numTuples, const sint p, const sint dim);
	void mergeSortRangeGPU(const sint start, const sint num, const sint from, const sint to,
			const sint p, const sint dim);
	sint removeDuplicatesGPU(const sint start, const sint num, const sint from, const sint to,
			const sint p, const sint dim, Gpu* otherGpu = NULL, sint otherNum = 0);
	refIdx_t buildKdTreeGPU(const sint numTuples, const int startP, const sint dim);
	void getKdNodesFromGPU(KdNode kdNodes[], const sint numTuples);
	void getReferenceFromGPU(refIdx_t reference[], const sint p, uint numTuples);
	void copyRefValGPU(sint start, sint num, sint from, sint to);
	void copyRefGPU(sint start, sint num, sint from, sint to);
	sint balancedSwapGPU(sint start, sint num, sint from, sint p, sint dim, Gpu* otherGpu);
	void swapMergeGPU(sint start, sint num, sint from, sint to, sint mergePoint,
			const sint p, const sint dim);
	void getCoordinatesFromGPU(KdCoord coord[],  const uint numTuples,  const sint dim);
	void fillMemGPU(uint* d_pntr, const uint val, const uint num);
	void fillMemGPU(sint* d_pntr, const sint val, const uint num);
	inline void syncGPU() { HIP_CHECK(hipStreamSynchronize(stream));}

private: // These are the methods specific mergeSort
	uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
	uint maxSampleCount;
	sint *d_mpi;  //This is where the per partition merge path data will get stored.
	refIdx_t* d_iRef;
	KdCoord* d_iVal;

	void mergeSortShared(
			KdCoord d_coords[],
			KdCoord  *d_DstVal,
			refIdx_t *d_DstRef,
			KdCoord  *d_SrcVal,
			refIdx_t *d_SrcRef,
			uint     batchSize,
			uint     arrayLength,
			uint     sortDir,
			sint      p,
			sint      dim
	);

	void generateSampleRanks(
			KdCoord d_coords[],
			uint     *d_RanksA,
			uint     *d_RanksB,
			KdCoord  *d_SrcVal,
			refIdx_t *d_SrcRef,
			uint     stride,
			uint     N,
			uint     sortDir,
			sint      p,
			sint      dim
	);

	void mergeRanksAndIndices(
			uint *d_LimitsA,
			uint *d_LimitsB,
			uint *d_RanksA,
			uint *d_RanksB,
			uint stride,
			uint N
	);

	void mergeElementaryInterRefs(
			KdCoord  d_coords[],
			KdCoord  *d_DstVal,
			refIdx_t *d_DstRef,
			KdCoord  *d_SrcVal,
			refIdx_t *d_SrcRef,
			uint     *d_LimitsA,
			uint     *d_LimitsB,
			uint     stride,
			uint     N,
			uint     sortDir,
			sint     p,
			sint     dim
	);

	void initMergeSortSmpl(uint N);
	void closeMergeSortSmpl();
	void mergeSortSmpl(
			KdCoord  d_coords[],
			KdCoord  *d_DstVal,
			refIdx_t *d_DstRef,
			KdCoord  *d_BufVal,
			refIdx_t *d_BufRef,
			KdCoord  *d_SrcVal,
			refIdx_t *d_SrcRef,
			uint     N,
			uint     sortDir,
			sint      p,
			sint      dim
	);
	uint balancedSwap(KdCoord* coordA, KdCoord* valA, refIdx_t* refA,
			KdCoord* coordB, KdCoord *valB, refIdx_t* refB,
			sint sortDir, sint p, sint dim, uint NperG, sint numThreads);

	void mergeSwap(KdCoord d_coord[], KdCoord d_valSrc[], refIdx_t d_refSrc[],
			KdCoord d_valDst[], refIdx_t d_refDst[],
			sint mergePnt, sint p, sint dim,  uint N, sint numThreads);

private: // These are the methods specific removeDupes
	uint copyRefVal(KdCoord valout[], refIdx_t refout[], KdCoord valin[], refIdx_t refin[], uint numTuples, sint numThreads);

	uint removeDups(KdCoord coords[], KdCoord val[], refIdx_t ref[], KdCoord valtmp[], refIdx_t reftmp[],
			KdCoord valin[], refIdx_t refin[], KdCoord otherCoord[], refIdx_t *otherRef,
			const sint p, const sint dim, const sint  numTuples, sint numThreads);

private: // These are the methods specific buildKdTree
	uint* d_segLengthsLT;  // When doing a segmented partition, pointers to the segment length arrays are stored here
	uint* d_segLengthsGT;
	refIdx_t* d_midRefs[2];

	void initBuildKdTree();
	void closeBuildKdTree();
	void partitionDim(KdNode d_kdNodes[], const KdCoord d_coords[], refIdx_t* l_references[],
			const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads);

	void partitionDimLast(KdNode d_kdNodes[], const KdCoord coord[], refIdx_t* l_references[],
			const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads);

	uint copyRef(refIdx_t refout[], refIdx_t refin[], uint numTuples, sint numThreads);

private: // These are the methods specific verifyKdTree
	// Pointer to the array for summing the node counts
	int* d_sums;
	void initVerifyKdTree();
	void closeVerifyKdTree();
	int  verifyKdTreeGPU(const sint root, const sint pstart, const sint dim, const sint numTuples);

};


#endif
