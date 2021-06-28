#include "includes.h"

#define MAX_CUDA_THREADS_PER_BLOCK 1024

__global__ void Max_Interleaved_Addressing_Shared(float* data, int data_size){
int idx = blockDim.x * blockIdx.x + threadIdx.x;
__shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
if (idx < data_size){

/*copy to shared memory*/
sdata[threadIdx.x] = data[idx];
__syncthreads();

for(int stride=1; stride < blockDim.x; stride *= 2) {
if (threadIdx.x % (2*stride) == 0) {
float lhs = sdata[threadIdx.x];
float rhs = sdata[threadIdx.x + stride];
sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
}
__syncthreads();
}
}
if (idx == 0) data[0] = sdata[0];
}