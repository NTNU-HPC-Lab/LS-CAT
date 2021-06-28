#include "includes.h"

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*2)


__global__ void stencil_1d(int *in, int *out)
{
__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;
int lindex = threadIdx.x + RADIUS;

temp[lindex] = in[gindex];
if (threadIdx.x < RADIUS)
{
temp[lindex - RADIUS] = in[gindex - RADIUS];
temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
}

__syncthreads();

int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
result += temp[lindex + offset];

out[gindex-RADIUS] = result;
}