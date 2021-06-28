#include "includes.h"
__global__ void vectorAdd(const uint16_t* A, const uint16_t* B, uint16_t* C, uint32_t numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
C[i] = A[i] + B[i];
}
}