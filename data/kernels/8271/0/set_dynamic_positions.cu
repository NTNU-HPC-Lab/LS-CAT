#include "includes.h"

__global__ void set_dynamic_positions(float *arr, float t)
{
int threadID = threadIdx.x;
int blockID = blockIdx.x;
int threads_per_block = blockDim.x;
int i = blockID * threads_per_block + threadID;

if (threadID == 0 or threadID == 1 or threadID == 2)
{
arr[i] = arr[i] * t;
}
}