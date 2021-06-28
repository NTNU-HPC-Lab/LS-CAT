#include "includes.h"
__global__ void matadd( int * m0, int * m1, std::size_t w, std::size_t h )
{
auto i = blockIdx.x * blockDim.x + threadIdx.x;
auto j = blockIdx.y * blockDim.y + threadIdx.y;
if( i < w && j < h )
m0[ i * w + j ] +=  m1[ i * w + j ];// i * w + j;
}