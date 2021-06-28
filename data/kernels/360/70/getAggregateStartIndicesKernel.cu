#include "includes.h"
__global__ void getAggregateStartIndicesKernel(int size, int *fineAggregateSort, int *aggregateRemapIndex)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
if(idx == 0 || fineAggregateSort[idx] != fineAggregateSort[idx - 1])
{
aggregateRemapIndex[fineAggregateSort[idx]] = idx;
}
}
}