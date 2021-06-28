#include "includes.h"
__global__ void addKernel(int * dev_a, int* x)
{
int i = threadIdx.x;
if (dev_a[i] < *x)
dev_a[i] = 0;
else
dev_a[i] = 1;
}