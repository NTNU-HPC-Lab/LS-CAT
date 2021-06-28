#include "includes.h"
__global__ void ReturnFloat( float *sum, float *out, const float *pIn )
{
out[threadIdx.x] = atomicAdd( &out[threadIdx.x], pIn[threadIdx.x] );
}