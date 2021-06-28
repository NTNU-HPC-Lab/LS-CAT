#include "includes.h"
__global__ void readWriteOffset(float *A, float *B, float *C, const int n, int offset)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = i + offset;

if (k < n) C[k] = A[k] + B[k];
}