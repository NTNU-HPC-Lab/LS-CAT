#include "includes.h"
__global__ void InvertValuesKernel(float *input, float* outputs, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if(id < size)
{
outputs[id] = 1.00f - input[id];
}
}