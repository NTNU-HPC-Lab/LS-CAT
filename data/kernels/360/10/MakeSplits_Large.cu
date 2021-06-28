#include "includes.h"
__global__ void MakeSplits_Large(int baseAggregateIdx, int* splitting, int* aggregation, int* aggMapAdjIndices, int* aggMapAdjacency, int* adjIndices, int* adjacency) {
int currentAgg = splitting[blockIdx.x];
int aggBegin = aggMapAdjIndices[currentAgg];
int aggSize = aggMapAdjIndices[currentAgg + 1] - aggBegin;
int newAgg = baseAggregateIdx + blockIdx.x;

__shared__ int nodeIds[256];
__shared__ int scratchA[256];
__shared__ int scratchB[256];
__shared__ int rootA;
__shared__ int rootB;
__shared__ int aCount, bCount;
__shared__ bool incomplete;
incomplete = true;
aCount = 1;
bCount = 1;

// Load in the node Id's from the aggregate map to the shared array:
if (threadIdx.x < aggSize)
nodeIds[threadIdx.x] = aggMapAdjacency[aggBegin + threadIdx.x];
__syncthreads();

// Each thread loads it's neighbors list into registers, translating into
// aggregate offsets.
int neighbors[40];
int nextNeighbor = 0;
int nodeId = -1;
if (threadIdx.x < aggSize)
{
nodeId = nodeIds[threadIdx.x];
int start = adjIndices[nodeId];
int end = adjIndices[nodeId + 1];
for (int i = start; i < end; i++)
{
int neighborId = adjacency[i];
int a = 0, b = aggSize - 1, midpoint;
while (a < b)
{
midpoint = a + ((b - a) / 2);
if (nodeIds[midpoint] < neighborId)
a = midpoint + 1;
else
b = midpoint;
}

if (nodeIds[a] == neighborId)
{
neighbors[nextNeighbor++] = a;
}
}
}
__syncthreads();

// Find the farthest node from the lowest indexed node (first root point)
// Start by marking the first node and threads without a node as seen
// Mark initial distances in scratch vector
if (threadIdx.x < aggSize)
scratchA[threadIdx.x] = threadIdx.x == 0 ? 0 : -1;
int myDist = threadIdx.x == 0 ? 0 : -1;
bool swapped = false;
incomplete = true;
__syncthreads();

while (incomplete)
{
// Set the incomplete flag to false
incomplete = false;
__syncthreads();

// Check if a neighbor has a positive distance
if (threadIdx.x < aggSize && myDist == -1)
{
for (int i = 0; i < nextNeighbor; i++)
{
int neighborDist = scratchA[neighbors[i]];
if (neighborDist > -1)
myDist = neighborDist + 1;
}
}
__syncthreads();

if (threadIdx.x < aggSize && myDist > 0 && !swapped)
{
swapped = true;
scratchA[threadIdx.x] = myDist;
rootA = threadIdx.x;
incomplete = true;
}
__syncthreads();
}

// Find the farthest node from the first root point (second root point)
// Start by marking the first node and threads without a node as seen
// Mark initial distances in scratch vector
if (threadIdx.x < aggSize)
scratchA[threadIdx.x] = threadIdx.x == rootA ? 0 : -1;
myDist = threadIdx.x == rootA ? 0 : -1;
swapped = false;
incomplete = true;
__syncthreads();

while (incomplete)
{
// Set the incomplete flag to false
incomplete = false;
__syncthreads();

// Check if a neighbor has a positive distance
if (threadIdx.x < aggSize && myDist == -1)
{
for (int i = 0; i < nextNeighbor; i++)
{
int neighborDist = scratchA[neighbors[i]];
if (neighborDist > -1)
{
myDist = neighborDist + 1;
}
}
}
__syncthreads();

if (threadIdx.x < aggSize && myDist > 0 && !swapped)
{
swapped = true;
scratchA[threadIdx.x] = myDist;
rootB = threadIdx.x;
incomplete = true;
}
__syncthreads();
}

// Setting an assigned aggregate label (In ScratchA) for every node with the node at
// rootA being assigned the current aggregate ID and the node at rootB
// being assigned the newAgg ID and set initial distances from a root node
// (In ScratchB) for each node, -1 for unknown and 0 for the roots
int myAggregate = -1;
if (threadIdx.x == rootA)
myAggregate = currentAgg;
if (threadIdx.x == rootB)
myAggregate = newAgg;
if (threadIdx.x < aggSize)
{
scratchA[threadIdx.x] = myAggregate;
scratchB[threadIdx.x] = myAggregate > -1 ? 0 : -1;
}
incomplete = true;
__syncthreads();

// Assign nodes to each aggregate until no unassigned nodes remain.
while (incomplete)
{
// Set the incomplete flag to false
incomplete = false;
__syncthreads();

if (threadIdx.x < aggSize && myAggregate == -1)
{
for (int i = 0; i < nextNeighbor; i++)
{
int neighborAgg = scratchA[neighbors[i]];
if (neighborAgg > -1)
{
myDist = scratchB[neighbors[i]] + 1;
myAggregate = neighborAgg;
}
}
if (myAggregate == -1)
incomplete = true;
if (myAggregate == newAgg)
atomicAdd((unsigned int*)&bCount, (unsigned)1);
if (myAggregate == currentAgg)
atomicAdd((unsigned int*)&aCount, (unsigned)1);
}
__syncthreads();

if (threadIdx.x < aggSize)
{
scratchA[threadIdx.x] = myAggregate;
scratchB[threadIdx.x] = myDist;
}
__syncthreads();
}

// If the split was uneven try to repair it
int sizeDifference = aCount > bCount ? aCount - bCount : bCount - aCount;
bool moveToA = aCount < bCount;
__shared__ int moved;
moved = 0;
int toMove = sizeDifference / 2;
incomplete = true;
__syncthreads();

while (incomplete && moved < toMove)
{
incomplete = false;
__syncthreads();
bool swapping = false;
int newDist = INT_MAX;
if (threadIdx.x < aggSize)
{
bool canSwap = moveToA ? myAggregate == newAgg : myAggregate == currentAgg;
bool borderNode = false;

// Check if this node has no same aggregate neighbors of higher distance
// and on a border
for (int i = 0; i < nextNeighbor; i++)
{
int neighborAgg = scratchA[neighbors[i]];
int neighborDist = scratchB[neighbors[i]];
if (neighborAgg == myAggregate && neighborDist > myDist)
canSwap = false;

if (neighborAgg != myAggregate)
{
if (neighborDist + 1 < newDist)
newDist = neighborDist + 1;
borderNode = true;
}
}

// If a node could swap see if it will
if (borderNode && canSwap && atomicAdd((unsigned int*)&moved, 1) < toMove)
{
swapping = true;
}
}
__syncthreads();
if (swapping)
{
int a = moveToA ? 1 : -1;
atomicAdd((unsigned int*)&bCount, -a);
atomicAdd((unsigned int*)&aCount, a);
scratchA[threadIdx.x] = moveToA ? currentAgg : newAgg;
scratchB[threadIdx.x] = newDist;
incomplete = true;
}
__syncthreads();
}

// Write out the values to the aggregation array
if (threadIdx.x < aggSize)
{
aggregation[nodeId] = scratchA[threadIdx.x];
}

}