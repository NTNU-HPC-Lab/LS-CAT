#include "includes.h"

#define getPos(a,k) (((a)>>(k-1))&1)

extern "C" {



}
__global__ void prefixSum(int * input_T, int * prefix_T, int * prefix_helper_T, int n, int k, int blockPower) {
__shared__ int tmp_T[1024];

for(int i = 0; i<blockPower; i++) {
if(threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x >= n) return;

tmp_T[threadIdx.x] = input_T[threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x];
tmp_T[threadIdx.x] = getPos(tmp_T[threadIdx.x],k);

int val,kk = 1;
while(kk <= 512) {
__syncthreads();
if(kk <= threadIdx.x) val = tmp_T[threadIdx.x - kk];
__syncthreads();
if(kk <= threadIdx.x) tmp_T[threadIdx.x] += val;
kk *= 2;
}

__syncthreads();

prefix_T[threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x] = tmp_T[threadIdx.x];

if(threadIdx.x == 1023 || threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x == n-1) prefix_helper_T[i*gridDim.x + blockIdx.x + 1] = tmp_T[threadIdx.x];
}
}