#include "includes.h"
//#define _SIZE_T_DEFINED


extern "C"
{
}
__global__ void ShuffleRGB(float* input, float* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (id < size)
{
//int index = id / 3 + (id % 3) * (size / 3);
output[id / 3 + (id % 3) * (size / 3)] = input[id];
}
}