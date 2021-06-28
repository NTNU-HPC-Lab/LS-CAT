#include "includes.h"
__global__ void reduce0(float *g_idata, float *g_odata, int N){
extern __shared__ float sdata[];
// each thread loads one element from global to shared mem
int tid = threadIdx.x;
int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
if((i+blockDim.x)<N)
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
else if(i<N)
sdata[tid] = g_idata[i];
else
sdata[tid] = 0.0;
__syncthreads();
// do reduction in shared mem
for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
if (tid < 32)
{
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}

// do reduction in shared mem

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}