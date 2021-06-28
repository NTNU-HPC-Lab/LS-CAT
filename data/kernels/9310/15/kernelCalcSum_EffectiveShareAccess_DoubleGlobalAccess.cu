#include "includes.h"
__global__ static void kernelCalcSum_EffectiveShareAccess_DoubleGlobalAccess(const int* dataArray, int arraySize, int* sum)
{
__shared__ extern int cache[];

int cacheIndex = threadIdx.x;

int arrayIndex1 = (int)(blockDim.x * blockIdx.x + threadIdx.x); // first element
int arrayIndex2 = arrayIndex1 + gridDim.x * blockDim.x;         // second element

cache[cacheIndex] = 0;

if (arrayIndex1 < arraySize)
{
cache[cacheIndex] += dataArray[arrayIndex1];
}

if (arrayIndex2 < arraySize)
{
cache[cacheIndex] += dataArray[arrayIndex2];
}

__syncthreads();

int blockSize = blockDim.x;
for (int offset = blockSize >> 1; offset > 0; offset >>= 1)
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