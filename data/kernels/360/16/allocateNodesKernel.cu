#include "includes.h"
__global__ void allocateNodesKernel(int size, int *adjIndexes, int *adjacency, int *partIn, int *partOut, int *aggregated) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
if (aggregated[idx] == 0)
{
int start = adjIndexes[idx];
int end = adjIndexes[idx + 1];

// Storage for possible aggregations.
int candidates[10];
int candidateCounts[10];
for (int i = 0; i < 10; i++)
{
candidates[i] = -1;
candidateCounts[i] = 0;
}

// Going through neighbors to aggregate:
for (int i = start; i < end; i++)
{
int candidate = partIn[adjacency[i]];
if (candidate != -1)
{
for (int j = 0; j < 10 && candidate != -1; j++)
{
if (candidates[j] == -1)
{
candidates[j] = candidate;
candidateCounts[j] = 1;
} else
{
if (candidates[j] == candidate)
{
candidateCounts[j] += 1;
candidate = -1;
}
}
}
}
}

// Finding the most adjacent aggregate and adding node to it:
int addTo = candidates[0];
int count = candidateCounts[0];
for (int i = 1; i < 10; i++)
{
if (candidateCounts[i] > count)
{
count = candidateCounts[i];
addTo = candidates[i];
}
}
partOut[idx] = addTo;
if (addTo != -1)
{
aggregated[idx] = 1;
}
}
}
}