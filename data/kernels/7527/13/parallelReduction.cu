#include "includes.h"
__global__ void parallelReduction(int *d_array , int numberOfElements, int elementsPerThread,int numberOfThreadsPerBlock,int numberOfBlocks,int *d_global)
{
int index = blockIdx.x * blockDim.x + threadIdx.x ;
int sum = 0;

int j=0;
for(int i=index;i<numberOfElements;i = i+(numberOfBlocks*numberOfThreadsPerBlock))
{
sum = sum + d_array[i];
j++;
}
extern __shared__ int d_blockMemmory[];
d_blockMemmory[threadIdx.x] = sum;
sum =0;
__syncthreads();


if(threadIdx.x == 0)
{
for(int i =0; i<numberOfThreadsPerBlock;i++)
{
sum = sum+ d_blockMemmory[i];
}
d_global[blockIdx.x] = sum;
}
}