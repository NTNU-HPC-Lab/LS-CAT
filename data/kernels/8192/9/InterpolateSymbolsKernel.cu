#include "includes.h"
__global__ void InterpolateSymbolsKernel( float *symbolVectors, int symbolOneId, int symbolTwoId, float weightOne, float weightTwo, float *resultSymbol, int symbolSize )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < symbolSize)
{
int symbolOneCellId = symbolOneId * symbolSize + threadId;
int symbolTwoCellId = symbolTwoId * symbolSize + threadId;

resultSymbol[threadId] = weightOne * symbolVectors[symbolOneCellId] + weightTwo * symbolVectors[symbolTwoCellId];
}

}