#include "includes.h"
__global__ void saxpy(int * a, int * b, int * c)
{
int tid = blockIdx.x * blockDim.x * threadIdx.x;

if (tid < N)
c[tid] = 2 * a[tid] + b[tid];
}