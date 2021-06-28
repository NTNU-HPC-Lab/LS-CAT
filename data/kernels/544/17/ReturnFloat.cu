#include "includes.h"
__global__ void ReturnFloat( float *sum, float *out, const float *pIn )
{
extern __shared__ float s[];
s[threadIdx.x] = pIn[threadIdx.x];
__syncthreads();
(void) atomicAdd( &s[threadIdx.x], *pIn );
__syncthreads();
out[threadIdx.x] = s[threadIdx.x];
}