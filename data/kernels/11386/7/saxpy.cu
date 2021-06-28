#include "includes.h"
__global__ void saxpy(float scalar, float * x, float * y)
{
// Determine our unique global thread ID, so we know which element to process
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if ( tid < N ) // Make sure we don't do more work than we have data!
y[tid] = scalar * x[tid] + y[tid];
}