#include "includes.h"
__global__ static void kernelCalcSum_EffectiveShareAccess(const int* dataArray, int arraySize, int* sum)
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

int blockSize = blockDim.x;
for (int offset = blockSize >> 1; offset > 0; offset >>= 1) // code in this for block is changed
{
if (cacheIndex < offset)
{
cache[cacheIndex] += cache[cacheIndex ^ offset];
}
__syncthreads();
}

if (cacheIndex == 0)
{
atomicAdd(sum, cache[0]);
}
}