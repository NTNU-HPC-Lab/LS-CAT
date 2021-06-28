#include "includes.h"
__global__ void stencil_1d(int *in, int *out) {
// within a block, threads share data via shared memory ("global memory")
// data is not visible to threads in other blocks
// use __shared__ to declare a var/array in shared memory

__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
// each thread processs one output element (blockDim.x elements per block)
int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;
int lindex = threadIdx.x + RADIUS;

// read input elements into shared memory
temp[lindex] = in[gindex];
if (threadIdx.x < RADIUS) {
temp[lindex - RADIUS] = in[gindex - RADIUS];
temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
}

// synchronize all threads in the block : ensure all data is available
__syncthreads();

// apply the stencil
int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
result += temp[lindex + offset];
}

// store the result
out[gindex-RADIUS] = result;
}