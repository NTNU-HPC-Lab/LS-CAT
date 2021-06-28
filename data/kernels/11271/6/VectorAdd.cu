#include "includes.h"
__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
// Get our global thread ID
int i = blockIdx.x*blockDim.x+threadIdx.x;

//for (i = 0; i < n; ++i) // replaced

// Make sure we do not go out of bounds
if (i < n)
c[i] = a[i] + b[i];
}