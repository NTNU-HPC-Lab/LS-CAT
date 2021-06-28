#include "includes.h"
__global__ void pow_array_gpu(float *a, int power, int array_size)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

float result=1;
if (idx<array_size)
{
for(int i=0; i<power; ++i)
result*=a[idx];
a[idx] = result;
}
}