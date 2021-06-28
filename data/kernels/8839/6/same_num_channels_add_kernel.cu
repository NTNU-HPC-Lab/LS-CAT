#include "includes.h"
__global__ void same_num_channels_add_kernel(float *data_l, float *data_r, float *result)
{
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

result[threadId] = data_l[threadId] + data_r[threadId];
result[threadId + 1] = data_l[threadId + 1] + data_r[threadId + 1];
}