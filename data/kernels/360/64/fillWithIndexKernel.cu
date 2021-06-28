#include "includes.h"
__global__ void fillWithIndexKernel(int size, int *array)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
array[idx] = idx;
}
}