#include "includes.h"
// filename: vsquare.cu
// a simple CUDA kernel to element multiply vector with itself

extern "C"   // ensure function name to be exactly "vsquare"
{
}
__global__ void ax(const int lengthC, const double a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthC)
{
c[i] = a*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}
}