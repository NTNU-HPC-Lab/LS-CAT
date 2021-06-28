#include "includes.h"
// filename: gax.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gax"
{
}
__global__ void vsquare(const double *a, double *c)
{
int i = threadIdx.x+blockIdx.x*blockDim.x;
double v = a[i];
c[i] = v*v;
}