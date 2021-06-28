#include "includes.h"
__global__ void transformation(size_t num_values, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;

if(index < num_values)
{
size_t dest_index = (index/ld_src)*ld_src + ((index%ld_src)%8)*ld_dest+ (index%ld_src)/8;
dest[dest_index] = src[index];
}
}