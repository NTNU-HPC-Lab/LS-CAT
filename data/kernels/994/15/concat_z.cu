#include "includes.h"
__global__ void concat_z(size_t sz, float_t* src, float_t* dest, float_t* z, size_t stride)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

if(index < sz)
{
if(index>=stride)
{
dest[index]=src[index-stride];
}
else
{
dest[index]=z[index];
}
}
}