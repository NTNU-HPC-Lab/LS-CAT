#include "includes.h"
__global__ void FindDesirableMerges(int size, int minSize, int maxSize, bool force, int* adjIndices, int* adjacency, int *partSizes, int* desiredMerges, int* merging) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Only evaluate if the aggregate is not marked as active (merging
// or no possible merges)
if (merging[idx] != 1)
{
// Check through all neighboring aggregates for most desirable
int currentSize = partSizes[idx];
int checkedNeighbors = 0;
float bestDesirability = 0;
int mostDesirable = -1;
int start = adjIndices[idx];
int end = adjIndices[idx + 1];
for (int i = start; i < end; i++)
{
int neighborAgg = adjacency[i];

// Only active neighbor aggregates should be looked at:
if (merging[neighborAgg] != 1)
{
checkedNeighbors++;
int neighborSize = partSizes[neighborAgg];

float desirability = 0;
desirability += currentSize < minSize ? minSize - currentSize : 0;
desirability += neighborSize < minSize ? minSize - neighborSize : 0;
int totalSize = currentSize + neighborSize;
if (totalSize > maxSize)
desirability *= force ? 1.0/(totalSize - maxSize) : 0;

// If this merge is the most desirable seen mark it
if (desirability > bestDesirability)
{
bestDesirability = desirability;
mostDesirable = neighborAgg;
}
}
}

if (mostDesirable == -1)
merging[idx] = 1;

if (currentSize < minSize && force && mostDesirable == -1)
printf("Aggregate %d is too small but found no merges! %d / %d neighbors checked.\n",idx, checkedNeighbors, end-start);

desiredMerges[idx] = mostDesirable;
}
}
}