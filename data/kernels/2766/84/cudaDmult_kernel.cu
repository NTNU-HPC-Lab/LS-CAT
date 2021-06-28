#include "includes.h"
__global__ void cudaDmult_kernel(unsigned int size, const double *x1, const double *x2, double *y)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
y[i] = x1[i] * x2[i];
}
}