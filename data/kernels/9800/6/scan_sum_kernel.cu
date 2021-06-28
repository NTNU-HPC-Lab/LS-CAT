#include "includes.h"
__global__ void scan_sum_kernel(unsigned int* input_vals, unsigned int pass, unsigned int * output, unsigned int* output_block, unsigned int size, unsigned int block_num) {
unsigned int tid = threadIdx.x;
unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
__shared__ unsigned int shared_input_vals[BLOCK_SIZE];
__shared__ unsigned int shared_output[BLOCK_SIZE];
if (mid >= size) {
shared_input_vals[tid] = 0xFFFFFFFF;
} else {
shared_input_vals[tid] = input_vals[mid];
}

__syncthreads();

if (tid == 0 || ((shared_input_vals[tid - 1] >> pass) & 0x01)) {
shared_output[tid] = 0;
} else {
shared_output[tid] = 1;
}

__syncthreads();

for (unsigned int i = 1; i < BLOCK_SIZE; i <<= 1) {
unsigned int val = 0;
if (tid >= i) {
val = shared_output[tid - i];
}
__syncthreads();
shared_output[tid] += val;
__syncthreads();
}

if (mid < size) {
output[mid] = shared_output[tid];
if ((mid == size - 1) || (tid == BLOCK_SIZE-1)) {
output_block[blockIdx.x] = shared_output[tid];
if (!((shared_input_vals[tid] >> pass) & 0x01)) {
//output_block[mid/BLOCK_SIZE] += 1;
output_block[blockIdx.x] += 1;
}
}
}
__syncthreads();
}