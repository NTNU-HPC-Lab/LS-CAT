#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
//blockIdx is the value of the block index for whichever block is running the code
int tid = blockIdx.x;//handle the data at this index
//blockIdx has 2 dimensions; x and y. We only need one dimension
if(tid < N)
c[tid] = a[tid] + b[tid];
}