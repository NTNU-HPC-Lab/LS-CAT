#include "includes.h"
__global__ void Iterate(int size, int *originIn, int *originOut, int *bestSeenIn, int *bestSeenOut, int *adjIndexes, int *adjacency) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
int bestSeen = bestSeenIn[idx];
int origin = originIn[idx];
if (bestSeen < 1000001)
{
int start = adjIndexes[idx];
int end = adjIndexes[idx + 1];

// Look at all the neighbors and take best values:
for (int i = start; i < end; i++)
{
int neighbor = adjacency[i];
int challenger = bestSeenIn[neighbor];
int challengerOrigin = originIn[neighbor];

if (challenger > 0 && challenger == bestSeen && challengerOrigin > origin)
{
origin = challengerOrigin;
}


if (challenger > bestSeen)
{
bestSeen = challenger;
origin = challengerOrigin;
}
}
}

// Write out the best values found
bestSeenOut[idx] = bestSeen;
originOut[idx] = origin;
}
}