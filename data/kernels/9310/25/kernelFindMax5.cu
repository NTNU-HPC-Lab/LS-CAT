#include "includes.h"
__global__ static void kernelFindMax5(const int* dataArray, int arraySize, int* maxVal)
{
__shared__ extern int cache[];

int cacheIndex = threadIdx.x;

int arrayIndex1 = (int)(blockDim.x * blockIdx.x + threadIdx.x);
int arrayIndex2 = arrayIndex1 + gridDim.x * blockDim.x;

cache[cacheIndex] = INT_MIN;

if (arrayIndex1 < arraySize)
{
cache[cacheIndex] = max(cache[cacheIndex] , dataArray[arrayIndex1]);
}

if (arrayIndex2 < arraySize)
{
cache[cacheIndex] = max(cache[cacheIndex] , dataArray[arrayIndex2]);
}

__syncthreads();

int blockSize = blockDim.x;
for (int offset = blockSize >> 1; offset > 32; offset >>= 1)
{
if (cacheIndex < offset)
{
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ offset]);
}
__syncthreads();
}

// ワープは32スレッド単位なので、スレッドIDが32未満になったところでループ内容を展開
if (threadIdx.x < 32)
{
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 32]);
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 16]);
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 8]);
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 4]);
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 2]);
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ 1]);
}

if (cacheIndex == 0)
{
atomicMax(maxVal, cache[0]);
}
}