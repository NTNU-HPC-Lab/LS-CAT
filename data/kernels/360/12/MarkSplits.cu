#include "includes.h"
__global__ void MarkSplits(int size, bool force, int minPartSize, int maxPartSize, int* partSizes, int* splitsToMake) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
int currentSize = partSizes[idx];
bool shouldSplit = currentSize > maxPartSize && (force || currentSize > minPartSize * 2);
splitsToMake[idx] = shouldSplit ? 1 : 0;
}
}