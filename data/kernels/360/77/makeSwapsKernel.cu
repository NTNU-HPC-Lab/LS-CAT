#include "includes.h"
__global__ void makeSwapsKernel(int size, int *partition, int *partSizes, int *nodeWeights, int *swap_to, int *swap_from, int *swap_index, float *desirability)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx == size - 1)
{
if(desirability[idx] > .1)
{
int swapTo = swap_to[idx];
int swapFrom = swap_from[idx];
int swapIndex = swap_index[idx];
int nodeWeight = nodeWeights[swapIndex];
partition[swapIndex] = swapTo;
atomicAdd(&partSizes[swapTo], nodeWeight);
atomicAdd(&partSizes[swapFrom], -nodeWeight);
//printf("Swapping node: %d, %d from part: %d, %d to part: %d, %d desirability: %f\n", swapIndex, nodeWeight, swapFrom, partSizes[swapFrom], swapTo, partSizes[swapTo], desirability[idx]);
}
}

else if(idx < size - 1)
{
if(desirability[idx] > .1 && swap_from[idx] != swap_from[idx + 1])
{
int swapTo = swap_to[idx];
int swapFrom = swap_from[idx];
int swapIndex = swap_index[idx];
int nodeWeight = nodeWeights[swapIndex];
partition[swapIndex] = swapTo;
atomicAdd(&partSizes[swapTo], nodeWeight);
atomicAdd(&partSizes[swapFrom], -nodeWeight);
//printf("Swapping node: %d, %d from part: %d, %d to part: %d, %d desirability: %f\n", swapIndex, nodeWeight, swapFrom, partSizes[swapFrom], swapTo, partSizes[swapTo], desirability[idx]);
}
}
}