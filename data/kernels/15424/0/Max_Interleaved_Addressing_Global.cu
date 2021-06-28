#include "includes.h"

#define MAX_CUDA_THREADS_PER_BLOCK 1024

__global__ void Max_Interleaved_Addressing_Global(float* data, int data_size){
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < data_size){
for(int stride=1; stride < data_size; stride *= 2) {
if (idx % (2*stride) == 0) {
float lhs = data[idx];
float rhs = data[idx + stride];
data[idx] = lhs < rhs ? rhs : lhs;
}
__syncthreads();
}
}
}