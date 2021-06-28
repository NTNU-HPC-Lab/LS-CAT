#include "includes.h"
__global__ void Return64( unsigned long long *sum, unsigned long long *out, const unsigned long long *pIn )
{
out[threadIdx.x] = atomicAdd( &sum[threadIdx.x], *pIn );
}