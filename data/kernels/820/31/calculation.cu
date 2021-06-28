#include "includes.h"
__global__ void calculation(    char *a, char *b, int *c, int constant, int vector_size ) {

int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

__shared__ char sharedDataA[block_size+2]; // border for the block are needed
char curr_b;

// Populate border
if (threadIdx.x == 0){
sharedDataA[0] = a[tid>0?tid-1:(vector_size-1)];

} else if (threadIdx.x == block_size - 1){
sharedDataA[block_size + 1] = a[tid<(vector_size-1)?tid+1:0];

} else if (tid == vector_size - 1){
sharedDataA[threadIdx.x + 2] = a[0];
}

// How can we avoid these ifs??? Tip: Padding
if (tid < vector_size){
// Populate shared data for A
sharedDataA[threadIdx.x+1] = a[tid];

// Bring data from B (no need for shared)
curr_b = b[tid];
}

__syncthreads();

// Perform calculation
if (tid < vector_size){
int output_c = (sharedDataA[threadIdx.x]-sharedDataA[threadIdx.x+2])*curr_b; //Use neighbors from shared data
output_c += sharedDataA[threadIdx.x+1]*constant;

// Write result
c[tid] = output_c;
}
}