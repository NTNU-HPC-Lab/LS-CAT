#include "includes.h"
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){
unsigned int tid = threadIdx.x;
unsigned int idx = (4 * blockIdx.x) * blockDim.x + threadIdx.x;

int *idata = g_idata + (4 * blockIdx.x) * blockDim.x;

if(idx + 3 * blockDim.x < n){
g_idata[idx] += g_idata[idx + blockDim.x];
g_idata[idx] += g_idata[idx + 2 * blockDim.x];
g_idata[idx] += g_idata[idx + 3 * blockDim.x];
}
__syncthreads();

for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
if(tid < stride)
idata[tid] += idata[tid + stride];
__syncthreads();
}

if(tid == 0) g_odata[blockIdx.x] = idata[0];
}