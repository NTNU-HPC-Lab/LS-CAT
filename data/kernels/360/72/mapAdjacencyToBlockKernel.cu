#include "includes.h"
__global__ void mapAdjacencyToBlockKernel(int size, int *adjIndexes, int *adjacency, int *adjacencyBlockLabel, int *blockMappedAdjacency, int *fineAggregate)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int begin = adjIndexes[idx];
int end = adjIndexes[idx + 1];
int thisBlock = fineAggregate[idx];

// Fill block labeled adjacency and block mapped adjacency vectors
for(int i = begin; i < end; i++)
{
int neighbor = fineAggregate[adjacency[i]];

if(thisBlock == neighbor)
{
adjacencyBlockLabel[i] = -1;
blockMappedAdjacency[i] = -1;
}
else
{
adjacencyBlockLabel[i] = thisBlock;
blockMappedAdjacency[i] = neighbor;
}
}
}
}