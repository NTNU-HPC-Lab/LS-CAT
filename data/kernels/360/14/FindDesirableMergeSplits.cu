#include "includes.h"
__global__ void FindDesirableMergeSplits(int size, int minSize, int maxSize, int desiredSize, int* adjIndices, int* adjacency, int* partSizes, int* desiredMerges, int* merging) {
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
bool currentOutSized = currentSize < minSize || currentSize > maxSize;
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
bool neighborOutSized = neighborSize < minSize || neighborSize > maxSize;
int totalSize = currentSize + neighborSize;
bool legalPair = (neighborOutSized || currentOutSized) && totalSize > minSize * 2 && totalSize < maxSize * 2;
float desirability = legalPair ? 1.0 / abs(desiredSize - (currentSize + neighborSize)) : 0;

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

desiredMerges[idx] = mostDesirable;
}
}
}