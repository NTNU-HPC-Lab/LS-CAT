#include "includes.h"
__global__ void cuda_sirt_pixels_kernel(int p, int nx, int dx, float* recon, const float* data)
{
int d0      = blockIdx.x * blockDim.x + threadIdx.x;
int dstride = blockDim.x * gridDim.x;

for(int d = d0; d < dx; d += dstride)
{
float sum = 0.0f;
for(int i = 0; i < nx; ++i)
sum += recon[d * nx + i];
float upd = data[p * dx + d] - sum;
for(int i = 0; i < nx; ++i)
recon[d * nx + i] += upd;
}
}