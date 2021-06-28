#include "includes.h"
__global__ void sum4(float *A, float *B, float *C, const int N)
{
int j;
int i = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
for (j=0; j < 4; j++)
if  (i < N) {
C[i] = A[i] + B[i];
i += blockDim.x * gridDim.x;
}
}