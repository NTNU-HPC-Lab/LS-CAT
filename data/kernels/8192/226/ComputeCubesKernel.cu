#include "includes.h"
__global__ void ComputeCubesKernel(  float *pointsCoordinates, float *vertexData, int quadOffset, float cubeSide, int *activityFlag, int textureWidth, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
float x = pointsCoordinates[threadId * 3];
float y = pointsCoordinates[threadId * 3 + 1];
float z = pointsCoordinates[threadId * 3 + 2];

float halfSide = 0.50f * cubeSide;
if(activityFlag[threadId] == 0)
{
halfSide = 0.00f;
}

int textureOffset = quadOffset + maxCells * 4 * 6 * 3;
float textureAbsLength = (float)(maxCells * textureWidth);


// BOTTOM SIDE
vertexData[quadOffset + threadId * 72] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 1] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 2] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 3] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 4] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 5] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 6] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 7] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 8] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 9] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 10] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 11] = z + halfSide;

vertexData[textureOffset + threadId * 48] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 1] = 0.00f;

vertexData[textureOffset + threadId * 48 + 2] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 3] = 1.00f;

vertexData[textureOffset + threadId * 48 + 4] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 5] = 1.00f;

vertexData[textureOffset + threadId * 48 + 6] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 7] = 0.00f;


// FRONT SIDE
vertexData[quadOffset + threadId * 72 + 12] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 13] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 14] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 15] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 16] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 17] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 18] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 19] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 20] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 21] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 22] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 23] = z + halfSide;



vertexData[textureOffset + threadId * 48 + 8] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 9] = 0.00f;

vertexData[textureOffset + threadId * 48 + 10] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 11] = 1.00f;

vertexData[textureOffset + threadId * 48 + 12] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 13] = 1.00f;

vertexData[textureOffset + threadId * 48 + 14] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 15] = 0.00f;



// LEFT SIDE
vertexData[quadOffset + threadId * 72 + 24] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 25] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 26] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 27] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 28] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 29] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 30] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 31] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 32] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 33] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 34] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 35] = z + halfSide;


vertexData[textureOffset + threadId * 48 + 16] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 17] = 0.00f;

vertexData[textureOffset + threadId * 48 + 18] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 19] = 1.00f;

vertexData[textureOffset + threadId * 48 + 20] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 21] = 1.00f;

vertexData[textureOffset + threadId * 48 + 22] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 23] = 0.00f;

// BACK SIDE
vertexData[quadOffset + threadId * 72 + 36] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 37] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 38] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 39] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 40] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 41] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 42] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 43] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 44] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 45] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 46] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 47] = z - halfSide;


vertexData[textureOffset + threadId * 48 + 24] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 25] = 0.00f;

vertexData[textureOffset + threadId * 48 + 26] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 27] = 1.00f;

vertexData[textureOffset + threadId * 48 + 28] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 29] = 1.00f;

vertexData[textureOffset + threadId * 48 + 30] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 31] = 0.00f;


// RIGHT SIDE
vertexData[quadOffset + threadId * 72 + 48] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 49] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 50] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 51] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 52] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 53] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 54] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 55] = y - halfSide;
vertexData[quadOffset + threadId * 72 + 56] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 57] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 58] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 59] = z + halfSide;

vertexData[textureOffset + threadId * 48 + 32] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 33] = 0.00f;

vertexData[textureOffset + threadId * 48 + 34] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 35] = 1.00f;

vertexData[textureOffset + threadId * 48 + 36] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 37] = 1.00f;

vertexData[textureOffset + threadId * 48 + 38] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 39] = 0.00f;


// UPPER SIDE
vertexData[quadOffset + threadId * 72 + 60] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 61] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 62] = z + halfSide;

vertexData[quadOffset + threadId * 72 + 63] = x - halfSide;
vertexData[quadOffset + threadId * 72 + 64] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 65] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 66] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 67] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 68] = z - halfSide;

vertexData[quadOffset + threadId * 72 + 69] = x + halfSide;
vertexData[quadOffset + threadId * 72 + 70] = y + halfSide;
vertexData[quadOffset + threadId * 72 + 71] = z + halfSide;



vertexData[textureOffset + threadId * 48 + 40] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 41] = 1.00f;

vertexData[textureOffset + threadId * 48 + 42] = (float)((threadId) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 43] = 0.00f;

vertexData[textureOffset + threadId * 48 + 44] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 45] = 0.00f;

vertexData[textureOffset + threadId * 48 + 46] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
vertexData[textureOffset + threadId * 48 + 47] = 1.00f;
}



}