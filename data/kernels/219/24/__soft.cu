#include "includes.h"
__global__ void __soft(float* y, const float* x, float T, int m)
{

unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
float x_e, y_e;

if(xIndex < m)
{
x_e = x[xIndex];
y_e = fmaxf(fabsf(x_e) - T, 0.f);
y[xIndex] = y_e / (y_e + T) * x_e;
}
}