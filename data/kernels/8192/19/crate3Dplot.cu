#include "includes.h"
__global__ void crate3Dplot(float* plotValues, float patchSize, int itemsX, int itemsY, float maxValue, float* vertexData)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

int size = itemsX * itemsY;
float texSizeX = 1.0f / itemsX;
float texSizeY = 1.0f / itemsY;

int patchX = threadId / itemsY;
int patchY = itemsY - (threadId % itemsY) - 1;

if (threadId < size)
{
float height = plotValues[threadId] / maxValue;
float gap = 0.000;

float3* vertTop = (float3*)vertexData;
float2* texCoords = (float2*)(vertexData + (60 * size));

//top side
vertTop[threadId * 4].x = patchX * patchSize + gap;
vertTop[threadId * 4].y = height;
vertTop[threadId * 4].z = patchY * patchSize + gap;

texCoords[threadId * 4].x = patchX * texSizeX;
texCoords[threadId * 4].y = patchY * texSizeY;

vertTop[threadId * 4 + 1].x = (patchX + 1) * patchSize - gap;
vertTop[threadId * 4 + 1].y = height;
vertTop[threadId * 4 + 1].z = patchY * patchSize + gap;

texCoords[threadId * 4 + 1].x = (patchX + 1) * texSizeX;
texCoords[threadId * 4 + 1].y = patchY * texSizeY;

vertTop[threadId * 4 + 2].x = (patchX + 1) * patchSize - gap;
vertTop[threadId * 4 + 2].y = height;
vertTop[threadId * 4 + 2].z = (patchY + 1) * patchSize - gap;

texCoords[threadId * 4 + 2].x = (patchX + 1) * texSizeX;
texCoords[threadId * 4 + 2].y = (patchY + 1) * texSizeY;

vertTop[threadId * 4 + 3].x = patchX * patchSize + gap;
vertTop[threadId * 4 + 3].y = height;
vertTop[threadId * 4 + 3].z = (patchY + 1) * patchSize - gap;

texCoords[threadId * 4 + 3].x = patchX * texSizeX;
texCoords[threadId * 4 + 3].y = (patchY + 1) * texSizeY;

float3* vertLeft = (float3*)(vertexData + 12 * size);

//left side
vertLeft[threadId * 4] = vertTop[threadId * 4];
vertLeft[threadId * 4].y = 0;
vertLeft[threadId * 4 + 1] = vertTop[threadId * 4];

vertLeft[threadId * 4 + 2] = vertTop[threadId * 4 + 3];
vertLeft[threadId * 4 + 3] = vertTop[threadId * 4 + 3];
vertLeft[threadId * 4 + 3].y = 0;

float3* vertFar = (float3*)(vertexData + 24 * size);

//far side
vertFar[threadId * 4] = vertTop[threadId * 4 + 2];
vertFar[threadId * 4].y = 0;
vertFar[threadId * 4 + 1] = vertTop[threadId * 4 + 3];
vertFar[threadId * 4 + 1].y = 0;

vertFar[threadId * 4 + 2] = vertTop[threadId * 4 + 3];
vertFar[threadId * 4 + 3] = vertTop[threadId * 4 + 2];

float3* vertNear = (float3*)(vertexData + 36 * size);

//near side
vertNear[threadId * 4] = vertTop[threadId * 4 + 1];
vertNear[threadId * 4].y = 0;
vertNear[threadId * 4 + 1] = vertTop[threadId * 4];
vertNear[threadId * 4 + 1].y = 0;

vertNear[threadId * 4 + 2] = vertTop[threadId * 4];
vertNear[threadId * 4 + 3] = vertTop[threadId * 4 + 1];

float3* vertRight = (float3*)(vertexData + 48 * size);

//right side
vertRight[threadId * 4] = vertTop[threadId * 4 + 2];
vertRight[threadId * 4].y = 0;
vertRight[threadId * 4 + 1] = vertTop[threadId * 4 + 2];

vertRight[threadId * 4 + 2] = vertTop[threadId * 4 + 1];
vertRight[threadId * 4 + 3] = vertTop[threadId * 4 + 1];
vertRight[threadId * 4 + 3].y = 0;
}
}