#include "includes.h"
__global__ void kernal2(int *A, int k, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numElements)
A[i] = A[i]*k;
}