#include "includes.h"
__global__ void __toInt(double *A, int *B, int N) {
int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
for (int i = ip; i < N; i += blockDim.x * gridDim.x * gridDim.y) {
B[i] = (int)(A[i]);
}
}