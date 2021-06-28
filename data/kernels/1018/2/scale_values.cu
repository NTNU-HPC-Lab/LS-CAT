#include "includes.h"
__global__ void scale_values(float *num, size_t size, float abs_max)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
num[idx] = (abs_max + abs_max) * num[idx] - abs_max;
}