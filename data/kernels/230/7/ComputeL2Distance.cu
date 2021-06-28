#include "includes.h"
__global__ void ComputeL2Distance(float *corrData, int numPts1) {
// Get the global point index, not the local index within our 16x16 chunk
const int p1 = blockIdx.x * 16 + threadIdx.x;
const int p2 = blockIdx.y * 16 + threadIdx.y;

// Make sure p1 and p2 are both within bounds
if (p1 < numPts1) {
const int idx = p1 * gridDim.y * 16 + p2;
if (corrData[idx] > -1) corrData[idx] = 2 - 2 * corrData[idx];
else corrData[idx] = FLT_MAX;
}
}