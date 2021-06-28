#include "includes.h"
// filename: vmult!.cu
// a simple CUDA kernel to element multiply two vectors C=alpha*A.*B

extern "C"   // ensure function name to be exactly "vmultbang"
{
}
__global__ void gaxpy4(const int n, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i < n) {
c[i] = (double) i;  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}

}