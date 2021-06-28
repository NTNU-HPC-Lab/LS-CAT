#include "includes.h"
__global__ void AddTo32( int *sum, int *out, const int *pIn )
{
(void) atomicAdd( &out[threadIdx.x], *pIn );
}