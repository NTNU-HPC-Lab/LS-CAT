#include "includes.h"
__device__ int f () { return 21; }
__global__ void AplusB_wait(int *ret, int a, int N, clock_t sleepInterval)
{
clock_t start = clock64();
while ( clock64() < start + sleepInterval ) { }
size_t gindex = threadIdx.x + blockIdx.x * blockDim.x;
if ( gindex < N ) // Only change the needed.
ret[gindex] = a + gindex;
}