#include "includes.h"
__global__ void AdaptWinningFractionKernel(  int s1, float *winningFraction, int *winningCount, float bParam, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
winningFraction[threadId] = winningFraction[threadId] + bParam * ((float)(threadId == s1) - winningFraction[threadId]);
winningCount[threadId] = winningCount[threadId] + (threadId == s1) * 1;
}
}