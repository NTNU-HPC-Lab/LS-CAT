#include "includes.h"
__global__ void gGather(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize, int offset) {
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if(idx >= sparseSize)
return;
if(sparseIndices[idx] >= -offset && sparseIndices[idx] + offset < denseSize)
sparseData[idx] = denseData[sparseIndices[idx] + offset];
}