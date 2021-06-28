#include "includes.h"
__global__ void kernel_calculate_sum(double * dev_array_sums, unsigned int array_size, double * dev_block_sums) {
//
// sum of input array
//
__shared__ double shared_sum[BLOCK_SIZE];

// each thread loads one element from global to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < array_size)
{
shared_sum[tid] = dev_array_sums[i];
}
else
{
shared_sum[tid] = 0;
}
//synchronize the local threads writing to the local memory cache
__syncthreads();
// do reduction in shared memory
for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
if (tid < s) {
shared_sum[tid] += shared_sum[tid + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0)
{
dev_block_sums[blockIdx.x] = shared_sum[0];
}
}