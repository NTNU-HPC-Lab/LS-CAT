#include "includes.h"
__global__ void dot_product(float *a, float *b, float *c)
{
c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}