#include "includes.h"
__global__ void CumulateThroughTimeKernel(float* memoryBlocks, int count, int sequenceLength)
{
int memoryIdx = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (memoryIdx < count)
{
for (size_t i = 1; i < sequenceLength; i++)
{
int memoryBlockOffset = i * count;
memoryBlocks[memoryIdx] += memoryBlocks[memoryBlockOffset + memoryIdx];
}
}
}