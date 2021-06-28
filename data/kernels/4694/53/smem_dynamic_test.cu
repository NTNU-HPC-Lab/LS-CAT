#include "includes.h"
__global__ void smem_dynamic_test(int * in, int * out, int size)
{
int tid = threadIdx.x;
int gid = blockIdx.x * blockDim.x + threadIdx.x;

extern __shared__ int smem[];

if (gid < size)
{
smem[tid] = in[gid];
out[gid] = smem[tid];
}
}