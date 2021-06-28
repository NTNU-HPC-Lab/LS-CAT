#include "includes.h"
__global__ void vecmabite( int *out, int *in, int threads, std::size_t size )
{
auto tid_x = threadIdx.x;
auto tid_b = blockIdx.x;
out[ tid_x  + threads * tid_b] = in[ 2 * (tid_x  + threads * tid_b) ];
}