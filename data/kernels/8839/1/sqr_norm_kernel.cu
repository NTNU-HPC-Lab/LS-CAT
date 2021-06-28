#include "includes.h"
__global__ void sqr_norm_kernel(int n, float *out, float *data, float rows, float cols)
{
extern __shared__ float sdata[];
int i = blockDim.x * threadIdx.y + threadIdx.x;
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

sdata[i] = 0;
sdata[i] = data[threadId] * data[threadId] + data[threadId + 1] * data[threadId + 1];
__syncthreads();

for (unsigned int s = (blockDim.x * blockDim.y + 1) / 2, old_s = blockDim.x * blockDim.y; s > 0; s >>= 1) {

if (old_s & 1) s += 1;

if (i < s && i + s < old_s) {
sdata[i] += sdata[i + s];
}
old_s = s;
__syncthreads();
}

if (i == 0) {
atomicAdd(&out[blockId / n], sdata[0] / (rows * cols));
}
}