#include "includes.h"
/*

blockIdx: block index
threadIdx: thread index within block
blockDim: threads per block (2)
gridDim: blocks per launch (N/2)

*/



#define N 10


__global__ void sum(int *a, int *b, int *c)
{
int i;
i = blockIdx.x * blockDim.x +
threadIdx.x;
c[i] = a[i] + b[i];
}