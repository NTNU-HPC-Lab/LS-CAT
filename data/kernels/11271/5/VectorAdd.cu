#include "includes.h"
__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
int i = blockIdx.x*blockDim.x+threadIdx.x;

//for (i = 0; i < n; ++i)
if (i < n)
c[i] = a[i] + b[i];
}