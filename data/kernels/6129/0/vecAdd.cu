#include "includes.h"

// CUDA kernel. Each thread takes care of one element of c

float* internalBuffer;
int nBuf;

__global__ void vecAdd(float *a, float *b, float *c, int n)
{
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n)
c[id] = a[id] + b[id];
}