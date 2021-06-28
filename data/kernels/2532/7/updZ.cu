#include "includes.h"
__global__ void updZ(float *z, float *f, float tz, float beta, int nx, int ny)
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
if (px<(nx - 1)) a = f[idx + 1] - f[idx];
if (py<(ny - 1)) b = f[idx + nx] - f[idx];

// update z

a = z[2 * idx + 0] + tz*a;
b = z[2 * idx + 1] + tz*b;

t = sqrtf(beta + a*a + b*b);
t = t<1. ? 1. : 1. / t;

z[2 * idx + 0] = a*t;
z[2 * idx + 1] = b*t;
}
}