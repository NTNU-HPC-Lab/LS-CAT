#include "includes.h"
__global__ void histogram_kernel(int* PartialHist, int* DeviceData, int DataCount,int* timer)
{
int tid = threadIdx.x;
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
clock_t start_atomic=0;
clock_t stop_atomic=0;

extern __shared__ int hist[];

if(tid==0)
{
start_atomic = clock();
}

for(int i = 0; i< H; i++)
hist[tid * H + i] = 0;

for(int j = gid; j < DataCount; j += stride)
hist[tid * H + DeviceData[j]]++;

__syncthreads();

for(int t_hist = 0; t_hist < blockDim.x; t_hist++)
{
atomicAdd(&PartialHist[tid],hist[t_hist * H + tid]);
atomicAdd(&PartialHist[tid + blockDim.x],hist[t_hist * H + tid + blockDim.x]);
}
stop_atomic=clock();

if(tid==0)
{
timer[blockIdx.x] = stop_atomic - start_atomic;
}
}