#include "includes.h"
__global__ void copyp2p( int4*         __restrict__  dest, int4   const* __restrict__  src, size_t                      num_elems)
{
size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
for (size_t i=globalId; i < num_elems; i+= gridSize)
{
dest[i] = src[i];
}
}