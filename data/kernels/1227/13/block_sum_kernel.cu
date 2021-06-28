#include "includes.h"
__global__ void block_sum_kernel(int *arr, int size, int *block_sums) {
int num_threads = blockDim.x * gridDim.x;
int tid = threadIdx.x + blockIdx.x * blockDim.x;

// Each thread finds local sum of its assigned area
int my_sum = 0;
__shared__ int smem[128];
while (tid < size) {
my_sum += arr[tid];
tid += num_threads;
}

smem[threadIdx.x] = my_sum;

// Barrier then use parallel reduction to get block sum
__syncthreads();
for (int i = blockDim.x / 2; i > 0; i /= 2) {
if (threadIdx.x < i) {
int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
smem[threadIdx.x] = temp;
}
__syncthreads();
}
// Block sum added to global arr
if (threadIdx.x == 0) {
block_sums[blockIdx.x] = smem[0];
}
}