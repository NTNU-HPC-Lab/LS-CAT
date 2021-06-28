#include "includes.h"
__global__ void cmax(float *d_in, float *max, int len)
{
extern __shared__ float smax[];

unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

smax[tid] = d_in[i]>d_in[i+len] ? d_in[i] : d_in[i+len];

__syncthreads();
if(blockDim.x > 512 && tid<512) {if(smax[tid] < smax[tid+512]) smax[tid] = smax[tid+512];}  __syncthreads();
if(blockDim.x > 256 && tid<256) {if(smax[tid] < smax[tid+256]) smax[tid] = smax[tid+256];}  __syncthreads();
if(blockDim.x > 128 && tid<128) {if(smax[tid] < smax[tid+128]) smax[tid] = smax[tid+128];}  __syncthreads();
if(blockDim.x > 64 && tid<64) {if(smax[tid] < smax[tid+64]) smax[tid] = smax[tid+64];}  __syncthreads();
if(tid<32) {
if(blockDim.x > 32 && smax[tid] < smax[tid+32]) smax[tid] = smax[tid+32];
if(blockDim.x > 16 && smax[tid] < smax[tid+16]) smax[tid] = smax[tid+16];
if(blockDim.x > 8 && smax[tid] < smax[tid+8]) smax[tid] = smax[tid+8];
if(blockDim.x > 4 && smax[tid] < smax[tid+4]) smax[tid] = smax[tid+4];
if(blockDim.x > 2 && smax[tid] < smax[tid+2]) smax[tid] = smax[tid+2];
if(smax[tid] < smax[tid+1]) smax[tid] = smax[tid+1];
__syncthreads();
}
if(tid == 0 )
{
max[blockIdx.x] = smax[0];
}
}