#include "includes.h"
__global__ void gpu_check(int threads, uint64_t *data, uint32_t *results, uint64_t target)
{
__shared__ uint32_t tmp[512/32];

int thread = (blockDim.x * blockIdx.x + threadIdx.x);

if(threadIdx.x < (512/32))
tmp[threadIdx.x] = 0;

__syncthreads();

if (thread < threads)
{
uint64_t highword = data[threads*3 + thread];
if(highword < target){
atomicOr(&tmp[threadIdx.x/32], 1 << (threadIdx.x%32));
}

__syncthreads();
if(threadIdx.x < (512/32))
results[blockIdx.x*(4096/32) + threadIdx.x] = tmp[threadIdx.x];
}
}