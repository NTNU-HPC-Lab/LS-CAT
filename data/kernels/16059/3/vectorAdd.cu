#include "includes.h"
__global__ void vectorAdd(uint16_t* A, const uint16_t* B, int32_t numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
A[i] += B[i];
}
}