#include "includes.h"
__device__ void warp_reduce(float* S,int tx){
S[tx] += S[tx + 32]; __syncthreads();
S[tx] += S[tx + 16]; __syncthreads();
S[tx] += S[tx + 8];  __syncthreads();
S[tx] += S[tx + 4];  __syncthreads();
S[tx] += S[tx + 2];  __syncthreads();
S[tx] += S[tx + 1];  __syncthreads();
}
__global__ void reduce_v5(float* in,float* out, int n){
int tx = threadIdx.x;
int bx = blockIdx.x;
int i  = bx*(BX*2)+tx;

__shared__ float S[BX];	//Want to have only BX amount of shared mem which is THREAD_MAX in previous

S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
__syncthreads();

if(BX >= 1024){                 // Max threads for block in my gpu is 1024
if(tx < 512)
S[tx] += S[tx+512];
__syncthreads();
}

if(BX >= 512){
if(tx < 256)
S[tx] += S[tx+256];
__syncthreads();
}

if(BX >= 256){
if(tx < 128)
S[tx] += S[tx+128];
__syncthreads();
}

if(BX >= 128){
if(tx < 64)
S[tx] += S[tx+64];
__syncthreads();
}

if(tx < WARP_SIZE) {				//WARP_SIZE is 32
warp_reduce(S,tx);				//Unroaling the last warp
}

if(tx==0)
out[bx] = S[0];
}