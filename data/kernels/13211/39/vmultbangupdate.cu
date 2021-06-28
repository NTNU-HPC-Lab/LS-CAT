#include "includes.h"
// filename: gax.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gax"
{
}
__global__ void vmultbangupdate(const int lengthA, const double alpha, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthA)
{
c[i] += alpha*a[i] * b[i];
}
}