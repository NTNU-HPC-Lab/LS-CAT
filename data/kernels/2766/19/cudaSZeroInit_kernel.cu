#include "includes.h"
__global__ void cudaSZeroInit_kernel(unsigned int size, float* data)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride)
data[i] = 0.0f;
}