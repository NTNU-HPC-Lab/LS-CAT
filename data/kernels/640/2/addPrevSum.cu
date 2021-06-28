#include "includes.h"




__global__ void addPrevSum(unsigned int* blkSumsScan, unsigned int* blkScans, unsigned int n)
{
int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
if (i < n)
{
blkScans[i] += blkSumsScan[blockIdx.x];
}
}