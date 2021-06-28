#include "includes.h"
__global__ void reduce_v3(float* in,float* out, int n){
int tx = threadIdx.x;
int bx = blockIdx.x;
int BX = blockDim.x; //same as THEAD_MAX
int i  = bx*(BX*2)+tx;

__shared__ float S[THEAD_MAX];

S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
__syncthreads();
for(int s=BX/2; s>0 ;s>>=1){
if(tx < s)
S[tx] += S[tx+s];
__syncthreads();
}
if(tx==0)
out[bx] = S[0];
}