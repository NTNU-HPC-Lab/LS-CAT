#include "includes.h"
__global__ void cudaDadd_kernel(unsigned int size, double value, const double *x, double *y)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
y[i] = x[i] + value;
}
}