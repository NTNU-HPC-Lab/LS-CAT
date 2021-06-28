#include "includes.h"
__global__ void transpose_kernel(size_t sz, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;
size_t i = index/ld_src, j= index%ld_src;
size_t dest_index = j*ld_dest + i;

if(index < sz)
{
dest[dest_index] = src[index];
}
}