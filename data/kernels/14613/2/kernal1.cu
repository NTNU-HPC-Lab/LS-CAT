#include "includes.h"
__global__ void kernal1(int *A, int *B, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numElements)
B[i] = A[i]+B[i];
}