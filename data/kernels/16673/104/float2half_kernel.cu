#include "includes.h"
__global__ void float2half_kernel(half *out, float *in)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

out[idx] = __float2half(in[idx]);
}