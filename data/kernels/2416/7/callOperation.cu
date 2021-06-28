#include "includes.h"
__global__ void callOperation(int *a, int *b, int *res, int x, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < n) {
res[tid] = a[tid] - (b[tid] * x);
}
}