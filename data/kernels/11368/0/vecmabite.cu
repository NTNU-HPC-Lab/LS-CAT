#include "includes.h"



__global__ void vecmabite( int *out, int *in, std::size_t size )
{
auto tid = threadIdx.x;
out[ tid ] = in[ 2 * tid ];
}