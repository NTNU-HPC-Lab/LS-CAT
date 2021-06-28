#include "includes.h"




__global__ void vectorAdd(const int *a, const int *b, int *c, int N)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

while(tid < N)
{
c[tid] = a[tid] + b[tid];
tid += blockDim.x * gridDim.x;
}
}