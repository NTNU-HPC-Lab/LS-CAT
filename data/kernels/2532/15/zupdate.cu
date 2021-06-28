#include "includes.h"
__global__ void zupdate(float *z, float *z0, float tau, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;
if (x<nx && y<ny)
{
float a = z[2 * idx + 0];
float b = z[2 * idx + 1];
float t = 1 / (1 + tau*sqrtf(a*a + b*b));
z[2 * idx + 0] = (z0[2 * idx + 0] + tau*z[2 * idx + 0])*t;
z[2 * idx + 1] = (z0[2 * idx + 1] + tau*z[2 * idx + 1])*t;
}
}