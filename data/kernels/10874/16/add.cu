#include "includes.h"
__global__ void add( int *a, int *b, int *c ) {
int tid = blockIdx.x;    // this thread handles the data at its thread id
if (tid < N)
c[tid] = a[tid] + b[tid];
}