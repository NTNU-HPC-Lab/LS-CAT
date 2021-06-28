#include "includes.h"
__global__ void vecadd( int * v0, int * v1, std::size_t size )
{
auto tid = blockIdx.x * blockDim.x + threadIdx.x;
if( tid < size )
{
v0[ tid ] += v1[ tid ];
}
}