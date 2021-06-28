#include "includes.h"
__global__ void zupdate2_dummy(float *z1, float *z2, float *f, float tau, int nx, int ny)
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
//		float fr=f[idx+1];
//		float fu=f[idx+nx];
//    if (!(px==(nx-1))) a = fr - fc;
//     if (!(py==(ny-1))) b = fu - fc;
a = fc;
b = fc;

// update z
t = 1 / (1 + tau*sqrtf(a*a + b*b));
z1[idx] = (z1[idx] + tau*a)*t;
z2[idx] = (z2[idx] + tau*b)*t;
}
}