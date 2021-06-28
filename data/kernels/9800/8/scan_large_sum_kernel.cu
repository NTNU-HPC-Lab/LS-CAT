#include "includes.h"
__global__ void scan_large_sum_kernel(unsigned int* output_block, unsigned int* output_val, unsigned int* output_pos, unsigned int* input_val, unsigned int* input_pos, unsigned int* histogram, unsigned int pass, unsigned int block_num, unsigned int size) {

__shared__ unsigned int shared_prefix_sum[BLOCK_SIZE];
unsigned int tid = threadIdx.x;
unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;

if (mid >= size) {
shared_prefix_sum[tid] = 1;
} else {
shared_prefix_sum[tid] = output_block[blockIdx.x] + output_val[mid];
}
//if (shared_prefix_sum[tid] >= size) printf("mid/BLOCK_SIZE=%d\n", mid/BLOCK_SIZE);
__syncthreads();


if (mid < size) {
unsigned int location = shared_prefix_sum[tid];
if ((input_val[mid] >> pass) & 0x01) {
location = mid + histogram[0] - shared_prefix_sum[tid];
}
if (location >= size) printf("pass=%d,input[mid]=%d,mid=%d, blockIdx.x=%d, histogram[0]=%d, shared_prefix_sum[tid]=%d\n",
pass, input_val[mid], mid, blockIdx.x, histogram[0], shared_prefix_sum[tid]);
output_val[mid] = location;
}
__syncthreads();
}