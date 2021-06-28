#include "includes.h"
__global__ void g_getSmrWgrad(float* wgrad, float* weight, float lambda, int len, int batch)
{
for(int i = 0; i < len; i += blockDim.x)
{
int id = i + threadIdx.x;
if(id < len)
{
wgrad[id] = lambda * weight[id] + wgrad[id] / batch;
}
}
}