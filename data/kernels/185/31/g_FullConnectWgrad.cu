#include "includes.h"
__global__ void g_FullConnectWgrad(float* wgrad, float* w, int len, float lambda, int batch)
{
for(int i = 0; i < len; i += blockDim.x * gridDim.x)
{
int id = i + blockDim.x * blockIdx.x + threadIdx.x;
if(id < len)
{
if(fabs(lambda) < 1e-10)
wgrad[id] = wgrad[id] / batch /** dropM[id]*/;
else
wgrad[id] = (wgrad[id] / batch + lambda * w[id]) /** dropM[id]*/;
}
}
}