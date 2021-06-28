#include "includes.h"
__global__ void inefficient_prefixSum(float* in, int in_length, float* out ){

//shared memory declaration
extern __shared__ float DSM[];

//compute index
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if(idx < in_length){
//load on shared memory
DSM[threadIdx.x] = in[idx];

//compute prefix_sum making sequence of sums
for(int stride = 1; stride <= threadIdx.x; stride *= 2){
__syncthreads();

DSM[threadIdx.x] =  DSM[threadIdx.x] + DSM[threadIdx.x - stride];
}

out[idx] = DSM[threadIdx.x];

}

}