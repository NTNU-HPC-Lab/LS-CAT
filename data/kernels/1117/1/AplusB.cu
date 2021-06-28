#include "includes.h"
__device__ int f () { return 21; }
__global__ void AplusB(int *ret, int a, int N)
{
size_t gindex = threadIdx.x + blockIdx.x * blockDim.x;
if ( gindex < N ) // Only change the needed.
ret[gindex] = a + gindex;
}