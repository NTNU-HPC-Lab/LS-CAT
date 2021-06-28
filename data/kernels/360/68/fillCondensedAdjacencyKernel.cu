#include "includes.h"
__global__ void fillCondensedAdjacencyKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *adjacencyOut, int *permutedAdjIndexesIn, int *permutedAdjacencyIn)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int oldBegin = permutedAdjIndexesIn[ aggregateIdx[idx] ];
int newBegin = adjIndexesOut[idx];
int runSize = adjIndexesOut[idx + 1] - newBegin;

// Copy adjacency over
for(int i = 0; i < runSize; i++)
{
adjacencyOut[newBegin + i] = permutedAdjacencyIn[oldBegin + i];
}
}
}