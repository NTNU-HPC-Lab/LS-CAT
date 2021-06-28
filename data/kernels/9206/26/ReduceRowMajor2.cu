#include "includes.h"
__global__ void ReduceRowMajor2(int *g_idata, int *g_odata, int size) {
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
sdata[tid] = 0;
if(i < size)
sdata[tid] = g_idata[i];
__syncthreads();
for(unsigned int s = 4; s < blockDim.x; s*=2) {
int index = 2*s*tid;
if(index < blockDim.x) {
sdata[index] += sdata[index+s];
sdata[index+1] += sdata[index+s+1];
sdata[index+2] += sdata[index+s+2];
sdata[index+3] += sdata[index+s+3];
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