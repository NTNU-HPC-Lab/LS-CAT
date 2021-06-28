#include "includes.h"
__global__ void cudaDSoftplus_propagate_kernel(double* x, double* y, unsigned int size)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
y[i] = log(1.0 + exp(x[i]));
}
}