#include "includes.h"
__global__ void VectorMultiplicationKernel(int *array, int arrayCount)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < arrayCount)
{
array[idx] *= array[idx];
}
}