#include "includes.h"
__global__ void addKernel(int *c, int *a,int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
//printf("%d", c[i]);
}