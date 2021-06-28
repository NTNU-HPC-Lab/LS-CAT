#include "includes.h"
__global__ void _negateStencilKernel(int* stencil, int size, int* out)
{
unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx >= size) return;

out[idx] = stencil[idx] == 1 ? 0 : 1;
}