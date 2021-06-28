#include "includes.h"
__global__ void MakeMerges_MarkSplits(int size, int* mergeWith, int* offsets, int* mis, int* splitsToMake) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
int currentAgg = mis[idx];
int newAgg = mergeWith[currentAgg];
// If the aggregate is not merging just apply offset
if (newAgg == -1)
{
mis[idx] = currentAgg - offsets[currentAgg];
}
// The aggregate is merging find offset of aggregate merging with
else
{
int newId = newAgg - offsets[newAgg];
mis[idx] = newId;
splitsToMake[newId] = 1;
}
}
}