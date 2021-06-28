#include "includes.h"
__global__ void zero_buffer( const int x, const int y, double* buffer)
{
const int gid = threadIdx.x+blockIdx.x*blockDim.x;

if(gid < x*y)
{
buffer[gid] = 0.0;
}
}