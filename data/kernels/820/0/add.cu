#include "includes.h"

#define block_size   32
#define vector_size  10000000


__global__ void add( int *a, int *b, int *c ) {
int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

if (tid < vector_size){
c[tid] = a[tid] + b[tid];                   // add vectors together
}
}