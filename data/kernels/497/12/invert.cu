#include "includes.h"
__global__ void invert(float *output, int* input, const int size)
{
int i = blockDim.x*blockIdx.x + threadIdx.x;
if(i<size)
{
int d = input[i];
if(d>0)
{
output[i] = __fdividef(1.f, d);
}
}
}