#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x; // handle the data at this index
while (tid < N)
{
c[tid] = a[tid] + b[tid];
tid += blockDim.x * gridDim.x;
}
}