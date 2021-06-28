#include "includes.h"
__global__ void addKernel(int * dev_a, int* dev_b ,int* dev_size)
{
int i = threadIdx.x;
int j,p;
for (j = 0; j < (*dev_size); j++)
{
p = *dev_size*i + j;
dev_b[i] += dev_a[p];
//printf("%d %d\n", i, p);
}
}