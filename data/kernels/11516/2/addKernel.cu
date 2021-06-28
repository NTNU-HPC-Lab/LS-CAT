#include "includes.h"
__global__ void addKernel(int * dev_a, int * dev_b, int * dev_c)
{
int i = threadIdx.x;
dev_c[i] = dev_a[i] + dev_b[i];
}