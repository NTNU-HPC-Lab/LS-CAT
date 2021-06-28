#include "includes.h"
__global__ void gpu_saxpy(int n, float a, float *x, float *y, float *s)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < n) s[i] = a*x[i] + y[i];
}