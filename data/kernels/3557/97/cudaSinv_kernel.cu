#include "includes.h"
__global__ void cudaSinv_kernel(unsigned int size, const float *x, float *y)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
y[i] = 1.0f / x[i];
}
}