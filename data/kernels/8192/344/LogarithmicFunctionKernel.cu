#include "includes.h"
__global__ void LogarithmicFunctionKernel(float* input, float* output, int size, const int type)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (id < size)
{
switch (type)
{
case 1:
output[id] = logf(input[id]);
break;
case 2:
output[id] = log2f(input[id]);
break;
case 3:
output[id] = log10f(input[id]);
break;
}
}
}