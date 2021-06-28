#include "includes.h"
__global__ void scales_channel_mul_kernel(float *data_l, float *data_r, float *result)
{
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x + blockIdx.x * blockDim.x * blockDim.y);

result[threadId] = data_l[threadId] * data_r[one_ch_index] - data_l[threadId + 1] * data_r[one_ch_index + 1];
result[threadId + 1] = data_l[threadId] * data_r[one_ch_index + 1] + data_l[threadId + 1] * data_r[one_ch_index];
}