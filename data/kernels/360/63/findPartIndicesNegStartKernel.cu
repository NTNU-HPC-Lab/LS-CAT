#include "includes.h"
__global__ void findPartIndicesNegStartKernel(int size, int *array, int *partIndices)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
if(idx < size)
{
int value = array[idx];
int nextValue = array[idx + 1];
if(value != nextValue)
partIndices[value + 1] = idx;
}
}