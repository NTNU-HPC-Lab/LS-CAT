#include "includes.h"
__global__ void hello2D()
{
int blocksize = blockIdx.y * blockIdx.x;
int blockId = gridDim.x * blockIdx.y + blockIdx.x;
int tid = blockId * blocksize + blockDim.x * threadIdx.y + threadIdx.x;

printf("I am thread (%d, %d) in block (%d, %d). Global thread ID = %d\n", threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, tid);
}