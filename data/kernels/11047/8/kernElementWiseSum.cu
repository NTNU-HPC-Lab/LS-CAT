#include "includes.h"
__device__ void devVecAdd(size_t pointDim, double* dest, double* src) {
for(size_t i = 0; i < pointDim; ++i) {
dest[i] += src[i];
}
}
__global__ void kernElementWiseSum(const size_t numPoints, const size_t pointDim, double* dest, double* src) {
// Called to standardize arrays to be a power of two

// Assumes a 2D grid of 1D blocks
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;

if(i < numPoints) {
devVecAdd(pointDim, &dest[i * pointDim], &src[i * pointDim]);
}
}