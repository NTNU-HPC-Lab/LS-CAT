#include "includes.h"
__global__ void dense_add(size_t sz, float_t* src, float_t* dest)
{
size_t srcIndex = threadIdx.x;
size_t destIndex = blockIdx.x*blockDim.x + threadIdx.x;
if(destIndex < sz)
{
dest[destIndex] += src[srcIndex];
}
}