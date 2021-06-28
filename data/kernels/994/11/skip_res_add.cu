#include "includes.h"
__global__ void skip_res_add(size_t sz, float_t* f5, float* f1, float_t* skip_out_sum, size_t stride)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

if(index < sz)
{
skip_out_sum[index] += f5[index+stride];
f1[index] += f5[index];
}
}