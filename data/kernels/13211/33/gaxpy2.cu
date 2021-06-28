#include "includes.h"
extern "C"
{
}
__global__ void gaxpy2(const double *a, const double *b, double *c)
{
int i = threadIdx.x + threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
c[i] = a[0]*b[i] + c[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}