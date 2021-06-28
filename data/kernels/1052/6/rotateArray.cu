#include "includes.h"
__global__ void rotateArray(int *c, int numThreads)
{
int nextIndex = (threadIdx.x + 1)%numThreads;
int val = c[nextIndex];

__syncthreads();

c[threadIdx.x] = val;
}