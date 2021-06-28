#include "includes.h"
__global__ void readOffsetUnroll4(float *A, float *B, float *C, const int n, int offset)
{
unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
unsigned int k = i + offset;

if (k + 3 * blockDim.x < n)
{
C[i]                  = A[k]                  + B[k];
C[i + blockDim.x]     = A[k + blockDim.x]     + B[k + blockDim.x];
C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
}
}