#include "includes.h"
__global__ void vecAdd_kernel(float *c, const float* a, const float* b)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = 0; i < 500; i++)
c[idx] = a[idx] + b[idx];
}