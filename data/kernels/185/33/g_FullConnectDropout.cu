#include "includes.h"
__global__ void g_FullConnectDropout(float * outputs, float * drop, int len)
{
for(int i = 0; i < len; i += blockDim.x * gridDim.x)
{
int id = i + blockIdx.x * blockDim.x + threadIdx.x;
if(id < len)
{
outputs[id] = outputs[id] * drop[id];
}
}
}