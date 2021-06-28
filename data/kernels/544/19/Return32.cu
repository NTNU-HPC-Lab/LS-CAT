#include "includes.h"
__global__ void Return32( int *sum, int *out, const int *pIn )
{
extern __shared__ int s[];
s[threadIdx.x] = pIn[threadIdx.x];
__syncthreads();
(void) atomicAdd( &s[threadIdx.x], *pIn );
__syncthreads();
out[threadIdx.x] = s[threadIdx.x];
}