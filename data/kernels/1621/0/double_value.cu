#include "includes.h"
__global__ void double_value(double *x, double *y)
{
y[threadIdx.x] = 2. * x[threadIdx.x];
}