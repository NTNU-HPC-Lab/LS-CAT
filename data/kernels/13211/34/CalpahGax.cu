#include "includes.h"
// filename: gaxpy2.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy2"
{
}
__global__ void CalpahGax(const double alpha, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
c[i] = alpha*a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}