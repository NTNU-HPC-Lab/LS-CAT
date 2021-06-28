#include "includes.h"
__global__ void FillOnes(float* vec, int value)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx > value) return ;

vec[idx] = 1.0f;
}