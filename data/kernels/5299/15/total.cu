#include "includes.h"
__global__ void total(float *input, float *output, int len){
__shared__ float partialSum[2*BLOCK_SIZE];
unsigned int t=threadIdx.x,start=2*blockIdx.x*BLOCK_SIZE;

if(start+t<len)	partialSum[t] = input[start+t];

else partialSum[t]=0;
__syncthreads();
if(start+BLOCK_SIZE+t<len)partialSum[BLOCK_SIZE+t]=input[start+BLOCK_SIZE+t];
else partialSum[BLOCK_SIZE+t]=0;
__syncthreads();
for(unsigned int stride=BLOCK_SIZE;stride>=1; stride>>=1){
__syncthreads();
if (t<stride) partialSum[t]+=partialSum[t+stride];
__syncthreads();
}
if(t==0) output[blockIdx.x]=partialSum[0];
}