#include "includes.h"

#define N 100000
#define THREAD_PER_BLOCK 1

/**
* This macro checks return value of the CUDA runtime call and exits
* the application if the call failed.
*/
__global__ void add(int *a, int *b, int *c) {
int tid = blockIdx.x; // handle the data at this index
if (tid < N) {
c[tid] = a[tid] + b[tid];
}
}