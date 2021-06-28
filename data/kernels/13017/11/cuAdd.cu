#include "includes.h"
__global__ void cuAdd(int *a,int *b,int *c, int N)
{
// 1D global index
int offset = blockDim.x * blockIdx.x + threadIdx.x;
if(offset < N)
{
c[offset] = a[offset] + b[offset];
}
}