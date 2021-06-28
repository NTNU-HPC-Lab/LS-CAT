#include "includes.h"
__global__ void ComputeBiasTermKernel(  float *biasTerm, float cFactor, float *winningFraction, int activeCells, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
biasTerm[threadId] = cFactor * ( 1.00f / activeCells - winningFraction[threadId]);
}
}