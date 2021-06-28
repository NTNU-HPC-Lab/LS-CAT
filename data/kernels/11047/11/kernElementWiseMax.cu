#include "includes.h"
__global__ void kernElementWiseMax(const size_t numPoints, double* dest, double* src) {
// Called to standardize arrays to be a power of two

// Assumes a 2D grid of 1D blocks
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;

if(i < numPoints) {
if(dest[i] < src[i]) {
dest[i] = src[i];
}
}
}