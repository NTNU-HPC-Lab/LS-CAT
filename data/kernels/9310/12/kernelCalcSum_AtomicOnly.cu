#include "includes.h"
__global__ static void kernelCalcSum_AtomicOnly(const int* dataArray, int arraySize, int* sum)
{
int arrayIndex = (int)(blockDim.x * blockIdx.x + threadIdx.x);
if (arrayIndex < arraySize)
{
atomicAdd(sum, dataArray[arrayIndex]);
}
}