#include "includes.h"
__global__ void FloorKernel(float* input, float* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
if(id < size)
{
output[id] = floor(input[id]);
}
}