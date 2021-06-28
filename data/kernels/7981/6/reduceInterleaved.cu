#include "includes.h"
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

int *idata = g_idata + blockIdx.x * blockDim.x;

if(idx >= n) return;

for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
if(tid < stride)
idata[tid] += idata[tid + stride];
__syncthreads();
}

if(tid == 0) g_odata[blockIdx.x] = idata[0];
}