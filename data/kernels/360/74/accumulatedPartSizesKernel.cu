#include "includes.h"
__global__ void accumulatedPartSizesKernel(int size, int *part, int *weights, int *accumulatedSize)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx == size - 1)
accumulatedSize[part[idx]] = weights[idx];
if(idx < size - 1)
{
int thisPart = part[idx];
if(thisPart != part[idx + 1])
accumulatedSize[thisPart] = weights[idx];
}
}