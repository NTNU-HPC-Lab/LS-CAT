#include "includes.h"
__global__ void fillPartitionLabelKernel(int size, int *coarseAggregate, int *fineAggregateSort, int *partitionLabel)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
partitionLabel[idx] = coarseAggregate[ fineAggregateSort[idx] ];
}
}