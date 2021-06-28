#include "includes.h"
__global__ void cudaSSoftplus_backPropagate_kernel(float* x, float* dx, unsigned int size)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
dx[i] *= (1.0f - exp(-x[i]));
}
}