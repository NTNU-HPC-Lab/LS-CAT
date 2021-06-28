#include "includes.h"
__global__ void sum(int *a, int *b, int *c, int N) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid<N) {
c[tid] = a[tid] + b[tid];
}
}