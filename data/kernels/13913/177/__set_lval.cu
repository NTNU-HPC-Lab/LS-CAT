#include "includes.h"
__global__ void __set_lval(long long *A, long long val, int length) {
int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
for (int i = ip; i < length; i += blockDim.x * gridDim.x * gridDim.y) {
A[i] = val;
}
}