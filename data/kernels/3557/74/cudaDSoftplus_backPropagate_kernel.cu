#include "includes.h"
__global__ void cudaDSoftplus_backPropagate_kernel(double* x, double* dx, unsigned int size)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
dx[i] *= (1.0 - exp(-x[i]));
}
}