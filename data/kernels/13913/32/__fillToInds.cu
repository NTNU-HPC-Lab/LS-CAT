#include "includes.h"
__global__ void __fillToInds(double A, double *B, int *I, long long len) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
int step = blockDim.x * gridDim.x * gridDim.y;
long long i;
for (i = tid; i < len; i += step) {
B[I[i]] = A;
}
}