#include "includes.h"
__global__ void getInducedGraphNeighborCountsKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *permutedAdjIndexes, int *permutedAdjacencyIn)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int Begin = permutedAdjIndexes[ aggregateIdx[idx] ];
int End = permutedAdjIndexes[ aggregateIdx[idx + 1] ];

// Sort each section of the adjacency:
for(int i = Begin; i < End - 1; i++)
{
for(int ii = i + 1; ii < End; ii++)
{
if(permutedAdjacencyIn[i] < permutedAdjacencyIn[ii])
{
int temp = permutedAdjacencyIn[i];
permutedAdjacencyIn[i] = permutedAdjacencyIn[ii];
permutedAdjacencyIn[ii] = temp;
}
}
}

// Scan through the sorted adjacency to get the condensed adjacency:
int neighborCount = 1;
if(permutedAdjacencyIn[Begin] == idx)
neighborCount = 0;

for(int i = Begin + 1; i < End; i++)
{
if(permutedAdjacencyIn[i] != permutedAdjacencyIn[i - 1] && permutedAdjacencyIn[i] != idx)
{
permutedAdjacencyIn[neighborCount + Begin] = permutedAdjacencyIn[i];
neighborCount++;
}
}

// Store the size
adjIndexesOut[idx] = neighborCount;
}
}