#include "includes.h"
__global__ void ComputeCubes2Kernel(  float *pointsCoordinates, float *vertexData, int quadOffset, float cubeSide, float *cubeOperation, float *cubeTexCoordinates, int *activityFlag, float textureWidth, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells * 6)
{
int cellId = threadId / 6;
float fCellId = (float)cellId;
int sideId = threadId % 6;

float x = pointsCoordinates[cellId * 3];
float y = pointsCoordinates[cellId * 3 + 1];
float z = pointsCoordinates[cellId * 3 + 2];

float halfSide = (activityFlag[cellId] == 1) * 0.50f * cubeSide;

int textureOffset = quadOffset + maxCells * 4 * 6 * 3;
float textureAbsLength = (float)maxCells * textureWidth;


vertexData[quadOffset + cellId * 72 + 12*sideId]     = x + operationMaskConstant[12*sideId] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 1] = y + operationMaskConstant[12*sideId + 1] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 2] = z + operationMaskConstant[12*sideId + 2] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 3] = x + operationMaskConstant[12*sideId + 3] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 4] = y + operationMaskConstant[12*sideId + 4] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 5] = z + operationMaskConstant[12*sideId + 5] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 6] = x + operationMaskConstant[12*sideId + 6] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 7] = y + operationMaskConstant[12*sideId + 7] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 8] = z + operationMaskConstant[12*sideId + 8] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 9]  = x + operationMaskConstant[12*sideId + 9] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 10] = y + operationMaskConstant[12*sideId + 10] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 11] = z + operationMaskConstant[12*sideId + 11] * halfSide;


vertexData[textureOffset + cellId * 48 + 8 * sideId]     = ((fCellId + cubeTexCoordinatesConstant[sideId * 8])* textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 1] = cubeTexCoordinatesConstant[sideId * 8 + 1];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 2] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 2]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 3] = cubeTexCoordinatesConstant[sideId * 8 + 3];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 4] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 4]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 5] = cubeTexCoordinatesConstant[sideId * 8 + 5];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 6] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 6]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 7] = cubeTexCoordinatesConstant[sideId * 8 + 7];


/*
vertexData[quadOffset + cellId * 72 + 12*sideId]     = x + cubeOperation[12*sideId] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 1] = y + cubeOperation[12*sideId + 1] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 2] = z + cubeOperation[12*sideId + 2] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 3] = x + cubeOperation[12*sideId + 3] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 4] = y + cubeOperation[12*sideId + 4] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 5] = z + cubeOperation[12*sideId + 5] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 6] = x + cubeOperation[12*sideId + 6] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 7] = y + cubeOperation[12*sideId + 7] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 8] = z + cubeOperation[12*sideId + 8] * halfSide;

vertexData[quadOffset + cellId * 72 + 12*sideId + 9]  = x + cubeOperation[12*sideId + 9] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 10] = y + cubeOperation[12*sideId + 10] * halfSide;
vertexData[quadOffset + cellId * 72 + 12*sideId + 11] = z + cubeOperation[12*sideId + 11] * halfSide;


vertexData[textureOffset + cellId * 48 + 8 * sideId]     = ((fCellId + cubeTexCoordinates[sideId * 8])* textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 1] = cubeTexCoordinates[sideId * 8 + 1];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 2] = ((fCellId + cubeTexCoordinates[sideId * 8 + 2]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 3] = cubeTexCoordinates[sideId * 8 + 3];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 4] = ((fCellId + cubeTexCoordinates[sideId * 8 + 4]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 5] = cubeTexCoordinates[sideId * 8 + 5];

vertexData[textureOffset + cellId * 48 + 8 * sideId + 6] = ((fCellId + cubeTexCoordinates[sideId * 8 + 6]) * textureWidth) / textureAbsLength;
vertexData[textureOffset + cellId * 48 + 8 * sideId + 7] = cubeTexCoordinates[sideId * 8 + 7];
*/
}
}