#include "includes.h"
__global__ void dot(int *a, int *b, int *c)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
while(i < N)
{
c[i] = a[i] * b[i];
i += blockDim.x * gridDim.x;
}
}