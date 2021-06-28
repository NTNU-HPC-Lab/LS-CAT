#include "includes.h"
__global__ void fill(int * m, std::size_t w ,  std::size_t h)
{
auto idx = blockIdx.x * blockDim.x + threadIdx.x;
auto idy = blockIdx.y * blockDim.y + threadIdx.y;


if( idx < w && idy <h )
{
m [ idy * w + idx ] = idy * w + idx;
}

}