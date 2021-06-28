#include "includes.h"
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
unsigned int tid = threadIdx.x;
unsigned int idx = (8 * blockIdx.x) * blockDim.x + threadIdx.x;

int *idata = g_idata + (8 * blockIdx.x) * blockDim.x;

if(idx + 7 * blockDim.x < n){
g_idata[idx] += g_idata[idx + blockDim.x];
g_idata[idx] += g_idata[idx + 2 * blockDim.x];
g_idata[idx] += g_idata[idx + 3 * blockDim.x];
g_idata[idx] += g_idata[idx + 4 * blockDim.x];
g_idata[idx] += g_idata[idx + 5 * blockDim.x];
g_idata[idx] += g_idata[idx + 6 * blockDim.x];
g_idata[idx] += g_idata[idx + 7 * blockDim.x];
}
__syncthreads();

if(blockDim.x >= 1024 && tid < 512)
idata[tid] += idata[tid + 512];
__syncthreads();
if(blockDim.x >= 512 && tid < 256)
idata[tid] += idata[tid + 256];
__syncthreads();
if(blockDim.x >= 256 && tid < 128)
idata[tid] += idata[tid + 128];
__syncthreads();
if(blockDim.x >= 128 && tid < 64)
idata[idx] += idata[tid + 64];
__syncthreads();

if(tid < 32){
volatile int *vmem = idata;
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
vmem[tid] += vmem[tid + 8];
vmem[tid] += vmem[tid + 4];
vmem[tid] += vmem[tid + 2];
vmem[tid] += vmem[tid + 1];
}

if(tid == 0) g_odata[blockIdx.x] = idata[0];
}