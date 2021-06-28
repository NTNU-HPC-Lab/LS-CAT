#include "includes.h"
__global__ void stencil_1d(int *in, int *out)
{
// blockDim is 3-dimensional vector storing block grid dimensions

// index of a thread across all threads + RADIUS
int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;

int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
result += in[gindex + offset];

// Store the result
out[gindex - RADIUS] = result;
}