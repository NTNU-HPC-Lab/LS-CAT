#include "includes.h"
__global__ void ReduceRowMajor(int *g_idata, int *g_odata, int size) {
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
sdata[tid] = 0;
if(i < size)
sdata[tid] = g_idata[i];
__syncthreads();
for(unsigned int s = 4; s < blockDim.x; s*=2) {
if(tid%(2*s) == 0) {
sdata[tid] += sdata[tid+s];
sdata[tid+1] += sdata[tid+s+1];
sdata[tid+2] += sdata[tid+s+2];
sdata[tid+3] += sdata[tid+s+3];
}
__syncthreads();
}
if(tid == 0) {
g_odata[blockIdx.x*4] = sdata[0];
g_odata[blockIdx.x*4+1] = sdata[1];
g_odata[blockIdx.x*4+2] = sdata[2];
g_odata[blockIdx.x*4+3] = sdata[3];
}
}