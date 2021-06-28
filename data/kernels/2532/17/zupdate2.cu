#include "includes.h"
__global__ void zupdate2(float *z, float *f, float tau, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
float a, b, t;

if (px<nx && py<ny)
{
// compute the gradient
a = 0;
b = 0;
float fc = f[idx];
if (!(px == (nx - 1))) a = f[idx + 1] - fc;
if (!(py == (ny - 1))) b = f[idx + nx] - fc;

// update z
t = 1 / (1 + tau*sqrtf(a*a + b*b));
z[2 * idx + 0] = (z[2 * idx + 0] + tau*a)*t;
z[2 * idx + 1] = (z[2 * idx + 1] + tau*b)*t;
}
}