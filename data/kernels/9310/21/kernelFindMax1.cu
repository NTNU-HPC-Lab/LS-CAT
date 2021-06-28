#include "includes.h"
__global__ static void kernelFindMax1(const int* dataArray, int arraySize, int* maxVal)
{
int arrayIndex = (int)(blockDim.x * blockIdx.x + threadIdx.x);
if (arrayIndex < arraySize)
{
atomicMax(maxVal, dataArray[arrayIndex]);
}
}