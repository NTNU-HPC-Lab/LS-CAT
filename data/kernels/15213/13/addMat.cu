#include "includes.h"
__global__ void addMat( float * mA_d, float * mB_d, std::size_t w, std::size_t h )
{
auto x = blockDim.x * blockIdx.x + threadIdx.x;
auto y = blockDim.y * blockIdx.y + threadIdx.y;

if( x < w && y < h )
{
mA_d[ y * w + x ] += mB_d[ y * w + x ];
}
}