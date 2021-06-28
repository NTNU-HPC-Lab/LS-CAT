#include "includes.h"
__global__ void dense_mv_add(size_t sz, float_t* src, float_t* dest)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;
if(index < sz)
{
dest[index] += src[index];
}
}