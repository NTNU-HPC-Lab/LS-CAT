#include "includes.h"
extern "C"
{
}
__global__ void gax(const int lengthC, const double *a, const double *b, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthC)
{
c[i] = a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
}
}