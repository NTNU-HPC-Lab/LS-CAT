#include "includes.h"
// filename: gaxpy2.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy2"
{
}
__global__ void vdivbang(const int lengthA, const double alpha, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthA)
{
c[i] = alpha*a[i] / b[i];
}
}