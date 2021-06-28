#include "includes.h"
__global__ void scan_kernel(unsigned int* output_block, unsigned int block_num) {
__shared__ unsigned int shared_output[BLOCK_SIZE];

if (threadIdx.x >= block_num || threadIdx.x == 0) {
shared_output[threadIdx.x] = 0x0;
}  else {
shared_output[threadIdx.x] = output_block[threadIdx.x - 1];
}
__syncthreads();

for (unsigned int i = 1; i < block_num; i <<= 1) {
unsigned int val = 0;
if (threadIdx.x >= i) {
val = shared_output[threadIdx.x - i];
}
__syncthreads();
shared_output[threadIdx.x] += val;
__syncthreads();
}

if (threadIdx.x < block_num) {
output_block[threadIdx.x] = shared_output[threadIdx.x];
}
__syncthreads();
}