#include "includes.h"
__global__ void d_MM(float *a, float *b, float *c, int wA, int wB, int hA)
{
// global index
int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // col
int gidy = blockDim.y * blockIdx.y + threadIdx.y;  // row

if(gidx < wB && gidy < hA)
{
float sum = 0.f;
for(int k=0; k<wA; k++)
{
// Multiply row of A by column of B
sum += a[gidy*wA + k] * b[k*wB +gidx];
}
c[gidy * wB + gidx] = sum;
}
}