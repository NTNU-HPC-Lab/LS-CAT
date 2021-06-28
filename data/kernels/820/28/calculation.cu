#include "includes.h"
__global__ void calculation(    int *a, int *b, int *c, int constant, int vector_size ) {

int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

if (tid < vector_size){

// Read in inputs
int prev_a = a[tid>0?tid-1:(vector_size-1)];
int curr_a = a[tid];
int post_a = a[tid<(vector_size-1)?tid+1:0];

int curr_b = b[tid];

// Do computation
int output_c = (prev_a-post_a)*curr_b + curr_a*constant;

// Write result
c[tid] = output_c;
}
}