#include "includes.h"
__global__ void maxKernel(float *array, int size, float* max)
{
extern __shared__ float sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
int stride = blockDim.x * 2 * gridDim.x;
sdata[tid] = 0;
while (i < size)
{
sdata[tid] = fmaxf(array[i], array[i + blockDim.x]);
i += stride;
__syncthreads();

}

for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
if (tid < s)
sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
__syncthreads();

}


if (tid < 32) {
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
__syncthreads();
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
__syncthreads();
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
__syncthreads();
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
__syncthreads();
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
__syncthreads();
sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
__syncthreads();

}
if (tid == 0) {
max[blockIdx.x] = sdata[0];
}
}