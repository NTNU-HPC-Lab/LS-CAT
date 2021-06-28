#include "includes.h"
__global__ void print_thread_variables()
{
printf("Thread{%d,%d,%d}, Block{%d,%d,%d}, BlockDim{%d,%d,%d}, GridDim{%d,%d,%d}\n",
threadIdx.x, threadIdx.y, threadIdx.z,
blockIdx.x, blockIdx.y, blockIdx.z,
blockDim.x, blockDim.y, blockDim.z,
gridDim.x, gridDim.y, gridDim.z
);
}