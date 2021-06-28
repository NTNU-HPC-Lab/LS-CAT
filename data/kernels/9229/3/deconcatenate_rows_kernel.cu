#include "includes.h"
__device__ float* deconcatenate(unsigned int x)
{
float * array = new float[32];

for (int i = 0; i < 32; i++)
{
array[i] = (x & ( 1 << i )) >> i;
}

return array;
}
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
__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size)
{
float * array;

for(int i=0; i<size; i+=32)
{
array = deconcatenate(a[i/32]);
for (int k=0;k<32;k++) b[i+k] = array[k];
delete[] array;
}
}