#include "includes.h"
__global__ void stencil(int *in, int *out)
{
int globIdx = blockIdx.x * blockDim.x + threadIdx.x;
int value = 0;
for(int offset = -RADIUS; offset <= RADIUS; offset++)
value += in[globIdx + offset];
out[globIdx] = value;
}