#include "includes.h"
__global__ void update_array_one_gpu(int m, int n, int i, int numberOfThreadsRequired,int count, int oldCount, int *d_array )
{
long j=blockIdx.x *blockDim.x + threadIdx.x;

if (j> numberOfThreadsRequired)
{}
else
{
d_Z1 = d_A1 + 1;
if (j < (m - 1) )
{
d_Z2 = d_A2 + 1;
}
}
}