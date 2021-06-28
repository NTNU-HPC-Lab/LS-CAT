#include "includes.h"
__global__ void writeLocalMemory(float *output, int size, int repeat)
{
int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
int tid=threadIdx.x, localSize=blockDim.x, litems=2048/localSize;
int s = tid;
__shared__ float lbuf[2048];
for (j=0 ; j<repeat ; ++j)
{
lbuf[(s+0)&(2047)] = gid;
lbuf[(s+1)&(2047)] = gid;
lbuf[(s+2)&(2047)] = gid;
lbuf[(s+3)&(2047)] = gid;
lbuf[(s+4)&(2047)] = gid;
lbuf[(s+5)&(2047)] = gid;
lbuf[(s+6)&(2047)] = gid;
lbuf[(s+7)&(2047)] = gid;
lbuf[(s+8)&(2047)] = gid;
lbuf[(s+9)&(2047)] = gid;
lbuf[(s+10)&(2047)] = gid;
lbuf[(s+11)&(2047)] = gid;
lbuf[(s+12)&(2047)] = gid;
lbuf[(s+13)&(2047)] = gid;
lbuf[(s+14)&(2047)] = gid;
lbuf[(s+15)&(2047)] = gid;
s = (s+16)&(2047);
}
__syncthreads();
for (j=0 ; j<litems ; ++j)
output[gid] = lbuf[tid];
}