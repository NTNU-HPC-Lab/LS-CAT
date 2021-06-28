#include "includes.h"
__global__ void cudaUZeroInit_kernel(unsigned int size, unsigned int* data)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride)
data[i] = 0U;
}