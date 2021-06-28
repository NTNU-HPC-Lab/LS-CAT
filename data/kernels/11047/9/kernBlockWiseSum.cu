#include "includes.h"
__global__ void kernBlockWiseSum(const size_t numPoints, const size_t pointDim, double* dest) {
// Assumes a 2D grid of 1024x1 1D blocks
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;

// call repeatedly for each dimension where dest is assumed to begin at dimension d

__shared__ double blockSum[1024];

if(threadIdx.x >= numPoints) {
blockSum[threadIdx.x] = 0;
} else {
blockSum[threadIdx.x] = dest[i * pointDim];
}

__syncthreads();

// Do all the calculations in block shared memory instead of global memory.
for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
blockSum[threadIdx.x] += blockSum[threadIdx.x + s];
__syncthreads();
}

if(threadIdx.x == 0) {
// Just do one global write
dest[i * pointDim] = blockSum[0];
}
}