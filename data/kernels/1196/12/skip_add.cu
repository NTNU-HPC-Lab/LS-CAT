#include "includes.h"
__global__ void skip_add(size_t sz, float_t* f1, float* skip_out_sum)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

if(index < sz)
{
skip_out_sum[index] += f1[index];
}
}