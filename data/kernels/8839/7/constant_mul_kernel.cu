#include "includes.h"
__global__ void constant_mul_kernel(float *data_l, float constant, float *result)
{
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

result[threadId] = data_l[threadId] * constant;
result[threadId + 1] = data_l[threadId + 1] * constant;
}