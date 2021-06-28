#include "includes.h"
__global__ void updhgZ_SoA(float *z1, float *z2, float *f, float tz, float lambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;

if (px<nx && py<ny)
{
// compute the gradient
float a = 0;
float b = 0;
float fc = f[idx];
if (px<(nx - 1)) a = f[idx + 1] - fc;
if (py<(ny - 1)) b = f[idx + nx] - fc;

// update z

a = z1[idx] + tz*lambda*a;
b = z2[idx] + tz*lambda*b;

// project
float t = 0;
t = sqrtf(a*a + b*b);
t = (t <= 1 ? 1. : t);

z1[idx] = a / t;
z2[idx] = b / t;
}
}