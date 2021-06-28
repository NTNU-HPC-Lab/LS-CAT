#include "includes.h"



__global__ void sqr_norm_kernel(const float *in, float *block_res, int total)
{
extern __shared__ float sdata[];
int in_idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
int i = threadIdx.x;
unsigned ins = blockDim.x;

if (in_idx >= total * 2)
sdata[i] = 0;
else
sdata[i] = in[in_idx] * in[in_idx] + in[in_idx + 1] * in[in_idx + 1];

for (unsigned outs = (ins + 1) / 2; ins > 1; ins = outs, outs = (outs + 1) / 2) {
__syncthreads();
if (i + outs < ins)
sdata[i] += sdata[i + outs];
}

if (i == 0)
block_res[blockIdx.x] = sdata[0];
}