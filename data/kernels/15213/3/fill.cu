#include "includes.h"
__global__ void fill( int * v, std::size_t size )
{
auto tid = threadIdx.x;
v[ tid ] = tid;
}