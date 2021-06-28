#include "includes.h"
__global__ void conj_kernel(float *data, float *result)
{
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

result[threadId] = data[threadId];
result[threadId + 1] = -data[threadId + 1];
}