#include "includes.h"
__global__ void matAdd(int *a, int *b, int *c)
{
const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * NUM_THREAD_IN_BLOCK);
int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

c[tID] = a[tID] + b[tID];
}