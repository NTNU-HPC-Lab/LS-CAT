#include "includes.h"
__global__ void remapAggregateIdxKernel(int size, int *fineAggregateSort, int *aggregateRemapId)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
fineAggregateSort[idx] = aggregateRemapId[fineAggregateSort[idx]];
}
}