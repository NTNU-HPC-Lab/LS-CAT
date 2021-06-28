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
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x;
if (index < N) {
intBuffer[index] = value;
}
}