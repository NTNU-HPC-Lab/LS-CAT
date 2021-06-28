#include "includes.h"


__global__ void rgb2yuv_gpu_son(unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , int size)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
if (x >= size) return;
unsigned char r, g, b;

r = d_r[x];
g = d_g[x];
b = d_b[x];

d_y[x] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
d_u[x] = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
d_v[x] = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
}