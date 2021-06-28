#include "includes.h"
#define GLM_FORCE_CUDA

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices, int *gridCellStartIndices, int *gridCellEndIndices) {
// TODO-2.1
// Identify the start point of each cell in the gridIndices array.
// This is basically a parallel unrolling of a loop that goes
// "this index doesn't match the one before it, must be a new cell!"
}