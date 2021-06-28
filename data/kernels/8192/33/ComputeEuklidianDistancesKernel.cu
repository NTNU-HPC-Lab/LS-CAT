#include "includes.h"
__global__ void ComputeEuklidianDistancesKernel(  float *inputImg, int imgWidth, int imgHeight, float *centroidCoordinates, float *distanceMatrix, int centroids, int inputSize )
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

float dist;

float centroidX;
float centroidY;

for(int c = 0; c < centroids; c++)
{
centroidX = centroidCoordinates[c * 2];
centroidY = centroidCoordinates[c * 2 + 1];

dist = sqrtf( (centroidX - X) * (centroidX - X) + (centroidY - Y) * (centroidY - Y) );
distanceMatrix[c * inputSize + threadId] = dist;
}
}
}