#include "includes.h"
__global__ void permuteInitialAdjacencyKernel(int size, int *adjIndexesIn, int *adjacencyIn, int *permutedAdjIndexesIn, int *permutedAdjacencyIn, int *ipermutation, int *fineAggregate)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int oldBegin = adjIndexesIn[ipermutation[idx]];
int oldEnd = adjIndexesIn[ipermutation[idx] + 1];
int runSize = oldEnd - oldBegin;
int newBegin = permutedAdjIndexesIn[idx];
//int newEnd = permutedAdjIndexesIn[idx + 1];
//int newRunSize = newEnd - newBegin;

//printf("Thread %d is copying from %d through %d into %d through %d\n", idx, oldBegin, oldEnd, newBegin, newEnd);

// Transfer old adjacency into new, while changing node id's with partition id's
for(int i = 0; i < runSize; i++)
{
permutedAdjacencyIn[newBegin + i] = fineAggregate[ adjacencyIn[oldBegin + i] ];
}
}
}