#include "includes.h"
__global__ void g_One_Bgrad(float* _delta, float* bgrad, int rows, int cols, int channels)
{
extern __shared__ float _sum[];
int channel = blockIdx.x;
int col     = blockIdx.y;
int row     = threadIdx.x;
float delta = _delta[channel * rows * cols + row * cols + col];
_sum[row] = delta;
__syncthreads();

int len = rows;
while(len != 1)
{
__syncthreads();
int skip = (len + 1) >> 1;
if(threadIdx.x < (len >> 1))
{
_sum[threadIdx.x] += _sum[threadIdx.x + skip];
}
len = (len + 1) >> 1;
}
__syncthreads();
if(threadIdx.x == 0)
{
bgrad[channel * cols + col] = _sum[0] / rows;
}
}