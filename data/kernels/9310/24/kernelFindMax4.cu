#include "includes.h"
__global__ static void kernelFindMax4(const int* dataArray, int arraySize, int* maxVal)
{
__shared__ extern int cache[];

int cacheIndex = threadIdx.x;

int arrayIndex1 = (int)(blockDim.x * blockIdx.x + threadIdx.x); // グローバルメモリの1つ目の要素番号
int arrayIndex2 = arrayIndex1 + gridDim.x * blockDim.x;         // グローバルメモリの2つ目の要素番号

cache[cacheIndex] = INT_MIN;

if (arrayIndex1 < arraySize)
{
cache[cacheIndex] = max(cache[cacheIndex] , dataArray[arrayIndex1]);    // シェアードメモリと比較
}

if (arrayIndex2 < arraySize)
{
cache[cacheIndex] = max(cache[cacheIndex] , dataArray[arrayIndex2]);    // シェアードメモリと比較
}

__syncthreads();

int blockSize = blockDim.x;
for (int offset = blockSize >> 1; offset > 0; offset >>= 1)
{
if (cacheIndex < offset)
{
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ offset]);
}
__syncthreads();
}

if (cacheIndex == 0)
{
atomicMax(maxVal, cache[0]);
}
}