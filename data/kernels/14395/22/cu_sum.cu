#include "includes.h"
__global__ void cu_sum(const float* src, float* sum, float *global_mem, const int n){
unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
// load input into __shared__ memory
float x = 0;
if(tid < n){
x = src[tid];
}
global_mem[threadIdx.x] = x;
__syncthreads();
// contiguous range pattern
for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
if(threadIdx.x < offset){
// add a partial sum upstream to our own
global_mem[threadIdx.x] += global_mem[threadIdx.x + offset];
}
// wait until all threads in the block have
// updated their partial sums
__syncthreads();
}
// thread 0 writes the final result
if(threadIdx.x == 0){
sum[blockIdx.x] = global_mem[0];
}
__syncthreads();
}