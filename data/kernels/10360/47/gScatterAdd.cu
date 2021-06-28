#include "includes.h"
__global__ void gScatterAdd(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize, int offset) {
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if(idx >= sparseSize)
return;
if(sparseIndices[idx] >= -offset && sparseIndices[idx] + offset < denseSize)
denseData[sparseIndices[idx] + offset] += sparseData[idx];
}