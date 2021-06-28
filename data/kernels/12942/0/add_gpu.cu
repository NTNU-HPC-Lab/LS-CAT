#include "includes.h"


#define N 10

__global__ void add_gpu( int *a, int *b, int *c ) {
int tid = blockIdx.x; // handle the data at this index
if (tid < N)
c[tid] = a[tid] + b[tid];
}