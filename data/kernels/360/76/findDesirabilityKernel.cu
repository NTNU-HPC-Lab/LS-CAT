#include "includes.h"
__global__ void findDesirabilityKernel(int size, int optimalSize, int *adjIndexes, int *adjacency, int *partition, int *partSizes, int *nodeWeights, int *swap_to, int *swap_from, int *swap_index, float *desirability)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
int currentPart = partition[idx];
int currentPartSize = partSizes[currentPart];
int nodeSize = nodeWeights[idx];
int selfAdjacency = 0;
int addTo = -1;
float bestDesirability = 0;

// The currentWeightFactor is higher the farther the count is from average
float currentWeightFactor = (float)abs(currentPartSize - optimalSize) / optimalSize;
// The self improvement is a measure of how much better this partitions size will be if the node is gone.
float selfImprovement = (abs(currentPartSize - optimalSize) - abs((currentPartSize - nodeSize) - optimalSize)) * currentWeightFactor;
if(selfImprovement > 0)
{
int start = adjIndexes[idx];
int end = adjIndexes[idx + 1];

// Arrays to store info about neighboring aggregates
int candidates[10];
int candidateCounts[10];
for(int i = 0; i < 10; i++)
{
candidates[i] = -1;
candidateCounts[i] = 0;
}

// Going through the neighbors:
for(int i = start; i < end; i++)
{
int candidate = partition[ adjacency[i] ];
if(candidate == currentPart)
selfAdjacency++;
else
for(int j = 0; j < 10; j++)
{
if(candidate != -1 && candidates[j] == -1)
{
candidates[j] = candidate;
candidateCounts[j] = 1;
candidate = -1;
}
else if(candidates[j] == candidate)
{
candidateCounts[j] += 1;
candidate = -1;
}
}
}

// Finding the best possible swap:
for(int i = 1; i < 10; i++)
{
if(candidates[i] != -1)
{
int neighborPart = candidates[i];
int neighborPartSize = partSizes[neighborPart];
float neighborWeightFactor = (float)abs(neighborPartSize - optimalSize) / optimalSize;
float neighborImprovement = ((float)(abs(neighborPartSize - optimalSize) - abs((neighborPartSize + nodeSize) - optimalSize))) * neighborWeightFactor;
// Combining with self improvement to get net
neighborImprovement += selfImprovement;
// Multiplying by adjacency factor
neighborImprovement *= (float)candidateCounts[i] / selfAdjacency;

if(neighborImprovement > bestDesirability)
{
addTo = neighborPart;
bestDesirability = neighborImprovement;
}
}
}
}

swap_from[idx] = currentPart;
swap_index[idx] = idx;
swap_to[idx] = addTo;
desirability[idx] = bestDesirability;
}
}