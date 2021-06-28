#include "includes.h"
__global__ void reduction(int* input, int* output) {
__shared__ int tmp[TPB];

tmp[threadIdx.x] = input[threadIdx.x + blockIdx.x * blockDim.x];

__syncthreads();

if(threadIdx.x < blockDim.x / 2)
tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 2];

__syncthreads();

if(threadIdx.x < blockDim.x / 4)
tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 4];

__syncthreads();

if(threadIdx.x < blockDim.x / 8)
tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 8];

__syncthreads();

if(threadIdx.x == 0) {
tmp[threadIdx.x] += tmp[threadIdx.x + 1];
output[blockIdx.x] = tmp[threadIdx.x];
}
}