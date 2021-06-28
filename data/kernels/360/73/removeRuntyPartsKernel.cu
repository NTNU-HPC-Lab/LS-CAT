#include "includes.h"
__global__ void removeRuntyPartsKernel(int size, int *partition, int *removeStencil, int *subtractions)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int currentNode = partition[idx];
if(removeStencil[currentNode] == 1)
partition[idx] = -1;
else
partition[idx] -= subtractions[currentNode];
}
}