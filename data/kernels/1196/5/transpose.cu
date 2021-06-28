#include "includes.h"
__global__ void transpose(size_t sz, float_t* src, float_t* dest, size_t src_width, size_t src_height)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

size_t i = index/src_width ;
size_t j = index%src_width;

size_t dest_index = j*src_height+i;
if(index < sz)
{
dest[dest_index] = src[index];
}
}