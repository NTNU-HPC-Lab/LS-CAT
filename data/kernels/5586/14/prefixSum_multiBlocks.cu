#include "includes.h"
__global__ void prefixSum_multiBlocks(float* in, int in_length, float* out, float* temp ){

extern __shared__ float DSM[];

int idx = blockIdx.x * blockDim.x + threadIdx.x;

//load in shared memory
if(idx < in_length){
DSM[threadIdx.x] = in[idx];

//partial sums phase
for(int stride = 1; stride <= blockDim.x; stride *= 2){
__syncthreads();
int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
if(index_aux < blockDim.x)
DSM[index_aux] += DSM[index_aux - stride];
}

//reduction phase
for(int stride=blockDim.x/4 ; stride > 0 ; stride /= 2){
__syncthreads();

int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
if(index_aux + stride < blockDim.x)
DSM[index_aux + stride] += DSM[index_aux];
}

__syncthreads();

//save intermediary values on temp to post combine for multi blocks
if(threadIdx.x == 0)
temp[blockIdx.x] = DSM[blockDim.x - 1];

out[idx] = DSM[threadIdx.x];

}

}