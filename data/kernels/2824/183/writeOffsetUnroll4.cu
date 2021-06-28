#include "includes.h"
__global__ void writeOffsetUnroll4(float *A, float *B, float *C, const int n, int offset)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = i + offset;

if (k + 3 * blockDim.x < n)
{
C[k]              = A[i]              + B[i];
C[k + blockDim.x]   = A[i +  blockDim.x] + B[i +  blockDim.x];
C[k + 2 * blockDim.x] = A[i + 2 * blockDim.x] + B[i + 2 * blockDim.x];
C[k + 3 * blockDim.x] = A[i + 3 * blockDim.x] + B[i + 3 * blockDim.x];
}
}