#include "includes.h"
__global__ void saxpy(int * a, int * b, int * c)
{
// Determine our unique global thread ID, so we know which element to process
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if ( tid < N ) // Make sure we don't do more work than we have data!
c[tid] = 2 * a[tid] + b[tid];
}