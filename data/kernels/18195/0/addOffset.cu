#include "includes.h"
__global__ void addOffset(int *dev_array, int length)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < length)
{
dev_array[tid] += OFFSET;
}
}