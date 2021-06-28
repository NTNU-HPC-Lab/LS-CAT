#include "includes.h"
__global__ void Return32( int *sum, int *out, const int *pIn )
{
out[threadIdx.x] = atomicAdd( &sum[threadIdx.x], *pIn );
}