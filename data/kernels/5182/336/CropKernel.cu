#include "includes.h"
__global__ void CropKernel(float min, float max, float* input, float* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if(id < size)
{
output[id] = fmaxf(fminf(input[id], max), min);
}
}