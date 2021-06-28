#include "includes.h"
__global__ void vectorAdd(int *A, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
A[i] = A[i] * 2;
}
}