#include "includes.h"
__global__ void ifpairmabite( int * v, std::size_t size )
{
// Get the id of the thread ( 0 -> 99 ).
auto tid = threadIdx.x;
// Each thread fills a single element of the array.
if (!(v[tid] % 2))
v[ tid ] *= 2;
}