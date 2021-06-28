#include "includes.h"
__global__ void normalize(int *values, int *max, float *output, int n)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < n)
{
output[i] = (float)values[i] / (float)*max;
}
}