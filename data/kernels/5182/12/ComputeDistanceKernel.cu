#include "includes.h"
__global__ void ComputeDistanceKernel( float *symbolVectors, float *inputVector, float *distance, int symbolSize, int symbols )
{
int symbolId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(symbolId < symbols)
{
float sum = 0.00f;
for(int i = 0; i < symbolSize; i++)
{
sum += symbolVectors[symbolId * symbolSize + i] * inputVector[i];
}
distance[symbolId] = sum;
}
}