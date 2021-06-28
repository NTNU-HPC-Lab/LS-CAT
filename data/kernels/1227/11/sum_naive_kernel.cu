#include "includes.h"
__global__ void sum_naive_kernel(int *arr, int size, int *sum) {
int num_threads = blockDim.x * gridDim.x;
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < size) {
atomicAdd(sum, arr[tid]);
tid += num_threads;
}
}