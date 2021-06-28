#include "includes.h"
__global__ void findPartIndicesKernel(int size, int *array, int *partIndices) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
int value = array[idx];
int nextValue = (idx != size - 1) ? array[idx + 1] : -1;
if (value != nextValue)
{
partIndices[value + 1] = idx + 1;
}
}
}