#include "includes.h"
__global__ void dense_add_conv(size_t sz, float_t* src, float_t* dest, size_t bias_dim)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;
// size_t src_index = index%bias_dim;
if(index < sz)
{
dest[index] += src[threadIdx.x];
}
}