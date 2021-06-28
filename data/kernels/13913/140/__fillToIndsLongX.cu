#include "includes.h"
__global__ void __fillToIndsLongX(long long A, long long *B, long long len) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
int step = blockDim.x * gridDim.x * gridDim.y;
long long i;
for (i = tid; i < len; i += step) {
B[i] = A;
}
}