#include "includes.h"
__global__ void zupdate_inter(float *z, float tau, int nx, int ny)
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
if (px<(nx - 1)) a = z[3 * (idx + 1) + 2] - z[3 * idx + 2];
if (py<(ny - 1)) b = z[3 * (idx + nx) + 2] - z[3 * idx + 2];

// update z
t = 1 / (1 + tau*sqrtf(a*a + b*b));
z[3 * idx + 0] = (z[3 * idx + 0] + tau*a)*t;
z[3 * idx + 1] = (z[3 * idx + 1] + tau*b)*t;
}
}