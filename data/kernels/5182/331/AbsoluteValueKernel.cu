#include "includes.h"
__global__ void AbsoluteValueKernel(float* input, float* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if(id < size)
{
output[id] = fabsf(input[id]);
}
}