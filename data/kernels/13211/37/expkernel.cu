#include "includes.h"
// filename: vsquare.cu
// a simple CUDA kernel to element multiply vector with itself

extern "C"   // ensure function name to be exactly "vsquare"
{
}
__global__ void expkernel(const int lengthA, const double *a,  double *b)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthA)
{
b[i] = exp(a[i]);
}
}