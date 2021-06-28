#include "includes.h"
__global__ void RemoveNodeByUtilityKernel(  int *connectionMatrix, int *connectionAge, int *activityFlag, float *utility, float utilityConstant, float *localError, int *neuronAge, float *winningFraction, int *winningCount, float maxError, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
if(activityFlag[threadId] == 1)
{
if(utility[threadId] > 0.00f)
{
if( maxError / utility[threadId] > utilityConstant )
{
activityFlag[threadId] = 0;
localError[threadId] = 0.00f;
neuronAge[threadId] = 0;
winningFraction[threadId] = 0.00f;
winningCount[threadId] = 0;
utility[threadId] = 0.00f;

for(int n = 0; n < maxCells; n++)
{
connectionMatrix[threadId * maxCells + n] = 0;
connectionAge[threadId * maxCells + n] = 0;
connectionMatrix[n * maxCells + threadId] = 0;
connectionAge[n * maxCells + threadId] = 0;
}
}
}
}
}
}