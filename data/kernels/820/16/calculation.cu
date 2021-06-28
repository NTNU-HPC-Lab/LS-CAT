#include "includes.h"
__global__ void calculation(    char *a, char *b, int *c, int constant, int vector_size ) {

int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

if (tid < vector_size){

// Read in inputs
char prev_a = a[tid>0?tid-1:(vector_size-1)];
char curr_a = a[tid];
char post_a = a[tid<(vector_size-1)?tid+1:0];

char curr_b = b[tid];

// Do computation
int output_c = (prev_a-post_a)*curr_b + curr_a*constant;

// Write result
c[tid] = output_c;
}
}