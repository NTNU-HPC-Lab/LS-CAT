#include "includes.h"
__global__ void AvgCentroidCoordinatesKernel(  float *centroidCoordinates, float *pointsWeight, int inputSize, int centroids )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < centroids * 2)
{
if(pointsWeight[threadId / 2] == 0.00f)
{
centroidCoordinates[threadId] = 0.00f;
}
else
{
centroidCoordinates[threadId] = centroidCoordinates[threadId] / pointsWeight[threadId / 2];
}
}
}