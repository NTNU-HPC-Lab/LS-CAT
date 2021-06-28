#include "includes.h"
__global__ void copy_kernel(size_t sz, float_t* src, float_t* dest)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

if(index < sz)
{
dest[index]=src[index];
}
}