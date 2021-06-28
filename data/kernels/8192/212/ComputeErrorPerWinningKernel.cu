#include "includes.h"
__global__ void ComputeErrorPerWinningKernel(  float *localError, int *winningCount, float *errorPerWinning, int *activityFlag, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;


// TO DO: GET RID OF IF-ELSE
if(threadId < maxCells)
{
if(activityFlag[threadId] == 1)
{
if(winningCount[threadId] != 0)
{
errorPerWinning[threadId] = localError[threadId] / (float)winningCount[threadId];
}
else
{
errorPerWinning[threadId] = 0.00f;
}
}
}
}