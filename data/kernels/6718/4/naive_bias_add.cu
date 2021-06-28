#include "includes.h"
__global__ void naive_bias_add(float *in, int size, float *bias, int bias_size)
{
int bid = blockIdx.x * blockDim.x + threadIdx.x;
if (!(bid < size)) return;
int bias_offset = bid - (bid / bias_size) * bias_size;
in[bid] += bias[bias_offset];
}