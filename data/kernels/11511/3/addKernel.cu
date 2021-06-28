#include "includes.h"
__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
int i = blockDim.x * blockIdx.x +  threadIdx.x;
if (i < size)
{
c[i] = a[i] + b[i];
}
}