#include "includes.h"
__global__ void cudaSSoftplus_propagate_kernel(float* x, float* y, unsigned int size)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
y[i] = log(1.0f + exp(x[i]));
}
}