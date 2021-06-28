#include "includes.h"
//Example 5.2.1. Pg 67, multiple block/threads
#define N (33 * 1024)



__global__ void add(int *a, int *b, int *c) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < N) {
c[tid] = a[tid] + b[tid];
tid += blockDim.x * gridDim.x;
}
}