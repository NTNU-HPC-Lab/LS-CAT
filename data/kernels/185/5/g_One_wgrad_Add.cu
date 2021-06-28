#include "includes.h"
__global__ void g_One_wgrad_Add( float* _WgradTmp, float* Wgrad, float* w, int rows, int cols, int channels, float lambda)
{
extern __shared__ float _sum[];
int channel = blockIdx.x;
int col     = blockIdx.y;
int tid     = threadIdx.x;
_sum[tid] = 0;
__syncthreads();


for(int i = 0; i < rows; i += blockDim.x){
int row = i + threadIdx.x;
if(row < rows){
_sum[threadIdx.x] += _WgradTmp[channel * rows * cols + row * cols + col];
}
}
__syncthreads();

int len = rows;
while(len != 1)
{
__syncthreads();
int skip = (len + 1) >> 1;
if(tid < (len >> 1))
{
_sum[tid] += _sum[tid + skip];
}
len = (len + 1) >> 1;
}
__syncthreads();
if(tid == 0)
{
Wgrad[channel * cols + col] = _sum[0] / rows + w[channel * cols + col] * lambda;
}
}