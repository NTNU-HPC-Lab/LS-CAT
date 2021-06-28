#include "includes.h"
__global__ void saxpy(int * a, int * b, int * c)
{
// Determine our unique global thread ID, so we know which element to process
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = tid; i < N; i += stride)
c[i] = 2 * a[i] + b[i];
}