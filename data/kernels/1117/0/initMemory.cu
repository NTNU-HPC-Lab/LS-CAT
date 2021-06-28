#include "includes.h"

__global__ void initMemory(size_t position, size_t* array)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
array[position + idx] = idx;
}