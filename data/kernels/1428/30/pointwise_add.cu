#include "includes.h"
__global__ void pointwise_add(float *d_res, const float *d_op1, const float *d_op2, const int len)
{
const int pos = blockIdx.x*blockDim.x + threadIdx.x;

if(pos >= len) return;

d_res[pos] = d_op1[pos] + d_op2[pos];
}