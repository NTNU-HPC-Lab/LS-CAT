#include "includes.h"
__global__ void print_unique_thread_id_1D()
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;

printf("Thread{%d,%d,%d}, Block{%d,%d,%d}, tid{%d}\n",
threadIdx.x, threadIdx.y, threadIdx.z,
blockIdx.x, blockIdx.y, blockIdx.z,
tid);
}