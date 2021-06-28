#include "includes.h"
__global__ void kernBlockWiseMax(const size_t numPoints, double* dest) {
// Assumes a 2D grid of 1024x1 1D blocks
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;

__shared__ double blockMax[1024];

if(threadIdx.x >= numPoints) {
blockMax[threadIdx.x] = -INFINITY;
} else {
blockMax[threadIdx.x] = dest[i];
}

__syncthreads();

// Do all the calculations in block shared memory instead of global memory.
for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
if(blockMax[threadIdx.x] < blockMax[threadIdx.x + s]) {
blockMax[threadIdx.x] = blockMax[threadIdx.x + s];
}
__syncthreads();
}

if(threadIdx.x == 0) {
// Just do one global write
dest[i] = blockMax[0];
}
}