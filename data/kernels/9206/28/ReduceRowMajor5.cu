#include "includes.h"
__device__ void warpReduce(volatile int* sdata, int tid, int n) {
if(tid + 32 < n)
sdata[tid] += sdata[tid+32];
if(tid + 16 < n)
sdata[tid] += sdata[tid+16];
if(tid + 8 < n)
sdata[tid] += sdata[tid+8];
if(tid + 4 < n)
sdata[tid] += sdata[tid+4];
}
__global__ void ReduceRowMajor5(int *g_idata, int *g_odata, int size) {
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
sdata[tid] = 0;
if(i < size)
sdata[tid] = g_idata[i];
__syncthreads();
for(unsigned int s = blockDim.x/2; s >= 32; s/=2) {
if(tid < s) {
sdata[tid] += sdata[tid+s];
}
__syncthreads();
}
if(tid < 32) {
warpReduce(sdata, tid, size);
}
if(tid == 0) {
g_odata[blockIdx.x*4] = sdata[0];
g_odata[blockIdx.x*4+1] = sdata[1];
g_odata[blockIdx.x*4+2] = sdata[2];
g_odata[blockIdx.x*4+3] = sdata[3];
}
}