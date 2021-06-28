#include "includes.h"
// filename: ax.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "ax"
{
}
__global__ void CalpahGaxpGy(const double alpha, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
c[i] = alpha*a[0]*b[i]+c[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}