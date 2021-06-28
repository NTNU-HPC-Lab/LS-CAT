#include "includes.h"
__device__ unsigned int concatenate(float* array)
{
unsigned int rvalue=0;
unsigned int sign;

for (int i = 0; i < 32; i++)
{
sign = (array[i]>=0);
rvalue = rvalue | (sign<<i);
}

return rvalue;
}
__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i<size) b[i] = concatenate(&a[i*32]);
}