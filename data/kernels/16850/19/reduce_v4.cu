#include "includes.h"
__device__ void warp_reduce(float* S,int tx){
S[tx] += S[tx + 32]; __syncthreads();
S[tx] += S[tx + 16]; __syncthreads();
S[tx] += S[tx + 8];  __syncthreads();
S[tx] += S[tx + 4];  __syncthreads();
S[tx] += S[tx + 2];  __syncthreads();
S[tx] += S[tx + 1];  __syncthreads();
}
__global__ void reduce_v4(float* in,float* out, int n){
int tx = threadIdx.x;
int bx = blockIdx.x;
int BX = blockDim.x; //same as THEAD_MAX
int i  = bx*(BX*2)+tx;

__shared__ float S[THEAD_MAX];

S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
__syncthreads();
for(int s=BX/2; s>WARP_SIZE ;s>>=1){
if(tx < s)
S[tx] += S[tx+s];
__syncthreads();
}
if(tx < WARP_SIZE)
warp_reduce(S,tx);				//Unroaling the last warp
if(tx==0)
out[bx] = S[0];
}