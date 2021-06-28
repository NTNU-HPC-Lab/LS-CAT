#include "includes.h"
__global__ void ReduceRowMajor3(int *g_idata, int *g_odata, int size) {
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
sdata[tid] = 0;
if(i < size)
sdata[tid] = g_idata[i];
__syncthreads();
for(unsigned int s = blockDim.x/2; s > 3; s/=2) {
if(tid < s) {
sdata[tid] += sdata[tid+s];
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