#include "includes.h"
__global__ void gpuSum(int *a, int *b, int *c, int n) {
int idx = threadIdx.x + (blockIdx.x * blockDim.x);
while (idx < n) {
c[idx] = a[idx] + b[idx];
idx += blockDim.x * gridDim.x;
}
}