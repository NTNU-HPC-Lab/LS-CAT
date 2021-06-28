#include "includes.h"
__global__ void stencil_1d(int *in, int *out){

__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
int gindex = threadIdx.x + blockIdx.x * blockDim.x;
int lindex = threadIdx.x + RADIUS;

// Debugging----------------------
//int *debug_sample = (int *)malloc(3*sizeof(int));

// Read input elements into shared memory
temp[lindex] = in[gindex + RADIUS]; // center

if (threadIdx.x < RADIUS) {
temp[threadIdx.x] = in[gindex]; // left
temp[lindex + BLOCK_SIZE] = in[gindex + RADIUS + BLOCK_SIZE]; // right
}

__syncthreads();

// Apply the stencil
int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++){
result += temp[lindex + offset];
//debug_sample[lindex + offset] = temp[lindex + offset];
}

//Debugging ---------------------
/*printf("Block %d, Thread %d"
" [%d, %d, %d]\n",blockIdx.x,threadIdx.x,
debug_sample[0],debug_sample[1],debug_sample[2]); */

// Store the result
out[gindex] = result;
}