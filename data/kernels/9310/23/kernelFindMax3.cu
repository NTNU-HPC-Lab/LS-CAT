#include "includes.h"
__global__ static void kernelFindMax3(const int* dataArray, int arraySize, int* maxVal)
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
cache[cacheIndex] = INT_MIN;
}

__syncthreads();

int blockSize = blockDim.x;
for (int offset = blockSize >> 1; offset > 0; offset >>= 1) // for文の中身を変更
{
if (cacheIndex < offset)
{
cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex ^ offset]); // オフセット計算も+からxorに変更（offsetは2の累乗値なのでxorにしても加算と同じになる）
}
__syncthreads();
}

if (cacheIndex == 0)
{
atomicMax(maxVal, cache[0]);
}
}