#include "includes.h"
__global__ void reductionKernel(float* vec, int width, double* sumUp){

//shared memory instantiation
extern __shared__ float partialSum[];

//index for global memory
int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
//index for shared memory
int b_idx = threadIdx.x;

//load shared memory from global memory
partialSum[b_idx] = g_idx < width ? vec[g_idx] : 0;

//reduction inside blocks
for(int stride = blockDim.x/2; stride >= 1 ; stride = stride/2){

__syncthreads();
if(b_idx < stride ){
partialSum[b_idx] = partialSum[b_idx] + partialSum[b_idx + stride];
}
}

//reduction for grid using just thread 0 of each block
if(b_idx == 0){
//coppy value back to global memory
vec[g_idx] = partialSum[b_idx];

//reduction
for(int stride = (gridDim.x * blockDim.x)/2; stride>=blockDim.x; stride = stride/2){

__syncthreads();
if(g_idx < stride){
vec[g_idx] = vec[g_idx] + vec[g_idx + stride];
}
}
}

//save result in output variable
if(g_idx == 0)
(*sumUp) = vec[g_idx];
}