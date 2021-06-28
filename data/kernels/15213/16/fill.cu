#include "includes.h"
__global__ void fill( int * v, std::size_t size )
{
// Get the id of the thread ( 0 -> 99 ).
auto tid = threadIdx.x;
// Each thread fills a single element of the array.
v[ tid ] = tid;
}