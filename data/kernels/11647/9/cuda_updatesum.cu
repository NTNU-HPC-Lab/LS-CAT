#include "includes.h"
__global__ void cuda_updatesum(int *array, int *update_array, int size)
{
extern __shared__ int shared[];

unsigned int tid = threadIdx.x;
unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
int op = 0;

if (blockIdx.x > 0) {
op = update_array[blockIdx.x - 1];
}

shared[tid] = array[id] + op;
array[id] = shared[tid];
}