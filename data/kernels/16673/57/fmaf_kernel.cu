#include "includes.h"
__global__ void fmaf_kernel(float *d_x, float *d_y, float *d_z, int size)
{
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = idx_x; i < size; i += stride) {
d_z[i] = fmaf(d_x[i], d_y[i], 0.f);
}
}