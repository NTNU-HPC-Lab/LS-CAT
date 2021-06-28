#include "includes.h"
__global__ void cmin(float *d_in, float *min, int len)
{
extern __shared__ float smin[];

unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

smin[tid] = d_in[i]<d_in[i+len] ? d_in[i] : d_in[i+len];

__syncthreads();
if(blockDim.x > 512 && tid<512) {if(smin[tid] > smin[tid+512]) smin[tid] = smin[tid+512];}  __syncthreads();
if(blockDim.x > 256 && tid<256) {if(smin[tid] > smin[tid+256]) smin[tid] = smin[tid+256];}  __syncthreads();
if(blockDim.x > 128 && tid<128) {if(smin[tid] > smin[tid+128]) smin[tid] = smin[tid+128];}  __syncthreads();
if(blockDim.x > 64 && tid<64) {if(smin[tid] > smin[tid+64]) smin[tid] = smin[tid+64];}  __syncthreads();
if(tid<32) {
if(blockDim.x > 32 && smin[tid] > smin[tid+32]) smin[tid] = smin[tid+32];
if(blockDim.x > 16 && smin[tid] > smin[tid+16]) smin[tid] = smin[tid+16];
if(blockDim.x > 8 && smin[tid] > smin[tid+8]) smin[tid] = smin[tid+8];
if(blockDim.x > 4 && smin[tid] > smin[tid+4]) smin[tid] = smin[tid+4];
if(blockDim.x > 2 && smin[tid] > smin[tid+2]) smin[tid] = smin[tid+2];
if(smin[tid] > smin[tid+1]) smin[tid] = smin[tid+1];
__syncthreads();
}

if(tid == 0 )
{
min[blockIdx.x] = smin[0];
}
}