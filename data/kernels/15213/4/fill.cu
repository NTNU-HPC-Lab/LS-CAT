#include "includes.h"
__global__ void fill(int * v, std::size_t size)
{
auto id = blockIdx.x * blockDim.x + threadIdx.x;

if( id < size)
{
v [ id ] = id;
}

}