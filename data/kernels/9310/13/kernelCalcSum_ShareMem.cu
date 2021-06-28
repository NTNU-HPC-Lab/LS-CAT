#include "includes.h"
__global__ static void kernelCalcSum_ShareMem(const int* dataArray, int arraySize, int* sum)
{
__shared__ extern int cache[];

int cacheIndex = threadIdx.x;

int arrayIndex = (int)(blockDim.x * blockIdx.x + threadIdx.x);
if (arrayIndex < arraySize)
{
cache[cacheIndex] = dataArray[arrayIndex];
}
else
{
cache[cacheIndex] = 0;
}

__syncthreads();

int baseIndex = threadIdx.x * 2;
int blockSize = blockDim.x;
for (int offset = 1; offset < blockSize; offset *= 2)
{
if (baseIndex + offset < blockSize)
{
cache[baseIndex] += cache[baseIndex + offset];
}
__syncthreads();
}

if (cacheIndex == 0)
{
atomicAdd(sum, cache[0]);
}
}