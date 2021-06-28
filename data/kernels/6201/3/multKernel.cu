#include "includes.h"
__global__ void multKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] * b[i] * 100;
}