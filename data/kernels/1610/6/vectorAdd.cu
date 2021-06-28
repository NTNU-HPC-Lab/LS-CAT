#include "includes.h"
__global__ void vectorAdd(int numElements, float *x, float *y)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
y[i] = x[i] + y[i];
}
}