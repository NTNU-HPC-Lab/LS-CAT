#include "includes.h"
__global__ void WinnersKernel(  float *winner, float *vertexData, int vertexOffset, float *pointsCoordinates, float cubeSize, int maxCells )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
if(winner[threadId] == 1.00f)
{
float x = pointsCoordinates[threadId * 3];
float y = pointsCoordinates[threadId * 3 + 1];
float z = pointsCoordinates[threadId * 3 + 2];

float side = 1.2f * cubeSize;
float halfSize = 0.50f * side;

// bottom side
vertexData[vertexOffset] = x - halfSize;
vertexData[vertexOffset + 1] = y - halfSize;
vertexData[vertexOffset + 2] = z - halfSize;

vertexData[vertexOffset + 3] = x - halfSize;
vertexData[vertexOffset + 4] = y - halfSize;
vertexData[vertexOffset + 5] = z + halfSize;


vertexData[vertexOffset + 6] = x + halfSize;
vertexData[vertexOffset + 7] = y - halfSize;
vertexData[vertexOffset + 8] = z + halfSize;

vertexData[vertexOffset + 9] = x + halfSize;
vertexData[vertexOffset + 10] = y - halfSize;
vertexData[vertexOffset + 11] = z - halfSize;
}
}
}