#include "includes.h"
__global__ void SumSymbolsKernel( float *symbolOne, float *symbolTwo, float *result, int symbolSize )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < symbolSize)
{
result[threadId] = symbolOne[threadId] + symbolTwo[threadId];
}
}