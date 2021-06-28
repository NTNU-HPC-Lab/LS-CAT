#include "includes.h"
__global__ void Finalize(int size, int *originIn, int *originOut, int *bestSeenIn, int *bestSeenOut, int *adjIndexes, int *adjacency, int *mis, int *incomplete) {
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
unsigned int challenger = bestSeenIn[neighbor];
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

// Write new MIS status
int misStatus = -1;
if (origin == idx)
misStatus = 1;
else if (bestSeen == 1000001)
misStatus = 0;

mis[idx] = misStatus;


// If this node is still unassigned mark
if (misStatus == -1)
{
incomplete[0] = 1;
}
}
}