#include "includes.h"
__global__ void readGlobalMemoryCoalesced(float *data, float *output, int size, int repeat)
{
int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
float sum = 0;
int s = gid;
for (j=0 ; j<repeat ; ++j)
{
float a0 = data[(s+0)&(size-1)];
float a1 = data[(s+32768)&(size-1)];
float a2 = data[(s+65536)&(size-1)];
float a3 = data[(s+98304)&(size-1)];
float a4 = data[(s+131072)&(size-1)];
float a5 = data[(s+163840)&(size-1)];
float a6 = data[(s+196608)&(size-1)];
float a7 = data[(s+229376)&(size-1)];
float a8 = data[(s+262144)&(size-1)];
float a9 = data[(s+294912)&(size-1)];
float a10 = data[(s+327680)&(size-1)];
float a11 = data[(s+360448)&(size-1)];
float a12 = data[(s+393216)&(size-1)];
float a13 = data[(s+425984)&(size-1)];
float a14 = data[(s+458752)&(size-1)];
float a15 = data[(s+491520)&(size-1)];
sum += a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
s = (s+524288)&(size-1);
}
output[gid] = sum;
}