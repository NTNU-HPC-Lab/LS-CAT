#include "includes.h"
__global__ void vecProductKernel(float *d_z, const float *d_x, const float *d_y, unsigned int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
d_z[idx] = d_x[idx] * d_y[idx];
}
}