#include "includes.h"
__global__ void saxpy(float *x, float *y, const float a)
{

const int i = blockIdx.x*blockDim.x + threadIdx.x;

if (i<ARRAY_SIZE) {
y[i] = a*x[i] + y[i];
}
}