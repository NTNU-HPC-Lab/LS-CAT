#include "includes.h"
__global__ void MarkMerges(int size, int* desiredMerges, int* merging, int* mergesToMake, int* incomplete) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Find what aggregate this one wants to merge with
int desiredMerge = desiredMerges[idx];

// If this aggregate has a real potential merger:
if (desiredMerge >= 0)
{
// If the aggregates agree to merge mark as merging
if (desiredMerges[desiredMerge] == idx)
{
// Mark the merge as the higher indexed aggregate merging into lower
if (desiredMerge > idx)
mergesToMake[desiredMerge] =  idx;
else
mergesToMake[idx] = desiredMerge;

// Mark both aggregates as merging
merging[idx] = 1;
merging[desiredMerge] = 1;
}
// Otherwise mark incomplete to check again
else
{
incomplete[0] = 1;
}
}
}
}