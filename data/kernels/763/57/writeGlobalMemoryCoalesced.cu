#include "includes.h"
__global__ void writeGlobalMemoryCoalesced(float *output, int size, int repeat)
{
int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
int s = gid;
for (j=0 ; j<repeat ; ++j)
{
output[(s+0)&(size-1)] = gid;
output[(s+32768)&(size-1)] = gid;
output[(s+65536)&(size-1)] = gid;
output[(s+98304)&(size-1)] = gid;
output[(s+131072)&(size-1)] = gid;
output[(s+163840)&(size-1)] = gid;
output[(s+196608)&(size-1)] = gid;
output[(s+229376)&(size-1)] = gid;
output[(s+262144)&(size-1)] = gid;
output[(s+294912)&(size-1)] = gid;
output[(s+327680)&(size-1)] = gid;
output[(s+360448)&(size-1)] = gid;
output[(s+393216)&(size-1)] = gid;
output[(s+425984)&(size-1)] = gid;
output[(s+458752)&(size-1)] = gid;
output[(s+491520)&(size-1)] = gid;
s = (s+524288)&(size-1);
}
}