#include "includes.h"
__global__ void reshape(size_t num_values, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;
if(index < num_values)
{
size_t src_index = (index/ld_dest)*ld_src+ index%ld_dest;
dest[index] = src[src_index];
}
}