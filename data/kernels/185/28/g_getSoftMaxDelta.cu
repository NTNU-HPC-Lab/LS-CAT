#include "includes.h"
__global__ void g_getSoftMaxDelta(float* softMaxDelta, float* softMaxP, float* groudTruth, int len)
{
for(int i = 0; i < len; i += blockDim.x)
{
int id = i + threadIdx.x;
if(id < len)
{
softMaxDelta[id] = softMaxP[id] - groudTruth[id];
}
}
}