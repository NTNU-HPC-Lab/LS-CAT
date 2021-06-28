#include "includes.h"
__global__ void AddToFloat( float *sum, float *out, const float *pIn )
{
(void) atomicAdd( &out[threadIdx.x], pIn[threadIdx.x] );
}