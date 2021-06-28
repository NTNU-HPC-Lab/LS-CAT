#include "includes.h"
__global__ void reduce_v0(float* in,float* out, int n){
int tx = threadIdx.x;
int bx = blockIdx.x;
int BX = blockDim.x; //same as THEAD_MAX
int i  = bx*BX+tx;

__shared__ float S[THEAD_MAX];

S[tx] = i < n ?  in[i] : 0;
__syncthreads();
for(int s=1; s<BX ;s*=2){
if(tx%(2*s)==0)
S[tx] += S[tx+s];
__syncthreads();
}
if(tx==0)
out[bx] = S[0];
}