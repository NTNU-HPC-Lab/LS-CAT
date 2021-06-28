#include "includes.h"
__global__ void TwoNodesDistanceKernel(  float *twoNodesDifference, float *twoNodesDistance, int vectorLength  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
float sum = 0.00f;
float value;
for(int i = 0; i < vectorLength; i++)
{
value = twoNodesDifference[threadId * vectorLength + i];
sum += value*value;
}
twoNodesDistance[threadId] = sqrtf(sum);
}
}