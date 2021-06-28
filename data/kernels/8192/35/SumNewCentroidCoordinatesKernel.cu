#include "includes.h"
__global__ void SumNewCentroidCoordinatesKernel( float *input, int imgWidth, int imgHeight, float *centroidCoordinates, int *nearestCentroid, float *pointsWeight, int inputSize )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
int pointX = threadId % imgWidth;
int pointY = threadId / imgWidth;

float X = (float)pointX;
float Y = (float)pointY;

int centroidId = nearestCentroid[threadId];

float weight = input[threadId];

atomicAdd(&centroidCoordinates[centroidId * 2], weight * X);
atomicAdd(&centroidCoordinates[centroidId * 2 + 1], weight * Y);
atomicAdd(&pointsWeight[centroidId], weight);
}
}