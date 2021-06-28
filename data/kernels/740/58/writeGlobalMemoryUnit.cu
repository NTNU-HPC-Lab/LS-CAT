#include "includes.h"
__global__ void writeGlobalMemoryUnit(float *output, int size, int repeat)
{
int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
int s = gid*512;
for (j=0 ; j<repeat ; ++j)
{
output[(s+0)&(size-1)] = gid;
output[(s+1)&(size-1)] = gid;
output[(s+2)&(size-1)] = gid;
output[(s+3)&(size-1)] = gid;
output[(s+4)&(size-1)] = gid;
output[(s+5)&(size-1)] = gid;
output[(s+6)&(size-1)] = gid;
output[(s+7)&(size-1)] = gid;
output[(s+8)&(size-1)] = gid;
output[(s+9)&(size-1)] = gid;
output[(s+10)&(size-1)] = gid;
output[(s+11)&(size-1)] = gid;
output[(s+12)&(size-1)] = gid;
output[(s+13)&(size-1)] = gid;
output[(s+14)&(size-1)] = gid;
output[(s+15)&(size-1)] = gid;
s = (s+16)&(size-1);
}
}