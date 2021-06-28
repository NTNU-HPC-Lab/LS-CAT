#include "includes.h"
__global__ void SumBasicSymbolsKernel( float *symbolVectors, int symbolOneId, int symbolTwoId, float *result, int symbolSize )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < symbolSize)
{
result[threadId] = symbolVectors[symbolOneId * symbolSize + threadId] + symbolVectors[symbolTwoId * symbolSize + threadId];
}
}