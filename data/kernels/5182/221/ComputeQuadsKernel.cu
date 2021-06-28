#include "includes.h"
__global__ void ComputeQuadsKernel(  float *pointsCoordinates, float *vertexData, int quadOffset, float textureSide, int *activityFlag, int textureWidth, int maxCells   )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
float x = pointsCoordinates[threadId * 3];
float y = pointsCoordinates[threadId * 3 + 1];
float z = pointsCoordinates[threadId * 3 + 2];

float halfSide = 0.50f * textureSide;
if(activityFlag[threadId] == 0)
{
halfSide = 0.00f;
}

int textureOffset = quadOffset + maxCells * 4 * 3 * 3;
float textureAbsLength = (float)(maxCells * textureWidth);

// vertical x-alligned
vertexData[quadOffset + threadId * 36] = x - halfSide;
vertexData[quadOffset + threadId * 36 + 1] = y + halfSide;
vertexData[quadOffset + threadId * 36 + 2] = z;

vertexData[textureOffset + threadId * 24] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 1] = 0.00f;

vertexData[quadOffset + threadId * 36 + 3] = x - halfSide;
vertexData[quadOffset + threadId * 36 + 4] = y - halfSide;
vertexData[quadOffset + threadId * 36 + 5] = z;

vertexData[textureOffset + threadId * 24 + 2] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 3] = 1.00f;

vertexData[quadOffset + threadId * 36 + 6] = x + halfSide;
vertexData[quadOffset + threadId * 36 + 7] = y - halfSide;
vertexData[quadOffset + threadId * 36 + 8] = z;

vertexData[textureOffset + threadId * 24 + 4] = (float)((threadId+1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 5] = 1.00f;


vertexData[quadOffset + threadId * 36 + 9] = x + halfSide;
vertexData[quadOffset + threadId * 36 + 10] = y + halfSide;
vertexData[quadOffset + threadId * 36 + 11] = z;

vertexData[textureOffset + threadId * 24 + 6] = (float)((threadId+1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 7] = 0.00f;

// horizontal
vertexData[quadOffset + threadId * 36 + 12] = x - halfSide;
vertexData[quadOffset + threadId * 36 + 13] = y;
vertexData[quadOffset + threadId * 36 + 14] = z + halfSide;

vertexData[textureOffset + threadId * 24 + 8] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 9] = 1.00f;

vertexData[quadOffset + threadId * 36 + 15] = x - halfSide;
vertexData[quadOffset + threadId * 36 + 16] = y;
vertexData[quadOffset + threadId * 36 + 17] = z - halfSide;

vertexData[textureOffset + threadId * 24 + 10] = (float)(threadId * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 11] = 0.00f;

vertexData[quadOffset + threadId * 36 + 18] = x + halfSide;
vertexData[quadOffset + threadId * 36 + 19] = y;
vertexData[quadOffset + threadId * 36 + 20] = z - halfSide;

vertexData[textureOffset + threadId * 24 + 12] = (float)((threadId+1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 13] = 0.00f;

vertexData[quadOffset + threadId * 36 + 21] = x + halfSide;
vertexData[quadOffset + threadId * 36 + 22] = y;
vertexData[quadOffset + threadId * 36 + 23] = z + halfSide;

vertexData[textureOffset + threadId * 24 + 14] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 15] = 1.00f;

// vertical z-alligned
vertexData[quadOffset + threadId * 36 + 24] = x;
vertexData[quadOffset + threadId * 36 + 25] = y - halfSide;
vertexData[quadOffset + threadId * 36 + 26] = z + halfSide;

vertexData[textureOffset + threadId * 24 + 16] = (float)((threadId+1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 17] = 1.00f;

vertexData[quadOffset + threadId * 36 + 27] = x;
vertexData[quadOffset + threadId * 36 + 28] = y - halfSide;
vertexData[quadOffset + threadId * 36 + 29] = z - halfSide;

vertexData[textureOffset + threadId * 24 + 18] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 19] = 1.00f;

vertexData[quadOffset + threadId * 36 + 30] = x;
vertexData[quadOffset + threadId * 36 + 31] = y + halfSide;
vertexData[quadOffset + threadId * 36 + 32] = z - halfSide;

vertexData[textureOffset + threadId * 24 + 20] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 21] = 0.00f;

vertexData[quadOffset + threadId * 36 + 33] = x;
vertexData[quadOffset + threadId * 36 + 34] = y + halfSide;
vertexData[quadOffset + threadId * 36 + 35] = z + halfSide;

vertexData[textureOffset + threadId * 24 + 22] = (float)((threadId+1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 24 + 23] = 0.00f;
}
}