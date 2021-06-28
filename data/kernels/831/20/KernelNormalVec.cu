#include "includes.h"
__global__ void KernelNormalVec(double *g_idata,double *g_odata,int l){ // Sequential Addressing technique

__shared__ double sdata[BLOCK_SIZE];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
if(i<l){ // bad thing -> severely punished performance.
sdata[tid] = g_idata[i];
}else{
sdata[tid] = 0.0;
}

__syncthreads();
// do reduction in shared mem
for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
if(tid < s){
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}