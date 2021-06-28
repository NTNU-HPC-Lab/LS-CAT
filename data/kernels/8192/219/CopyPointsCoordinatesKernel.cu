#include "includes.h"
__global__ void CopyPointsCoordinatesKernel(  float *pointsCoordinates, int *activityFlag, float xNonValid, float yNonValid, float zNonValid, float *dataVertex, int dataVertexOffset, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
float xToCopy = pointsCoordinates[threadId * 3];
float yToCopy = pointsCoordinates[threadId * 3 + 1];
float zToCopy = pointsCoordinates[threadId * 3 + 2];
if(activityFlag[threadId] == 0)
{
xToCopy = xNonValid;
yToCopy = yNonValid;
zToCopy = zNonValid;
}
dataVertex[dataVertexOffset + threadId * 3] = xToCopy;
dataVertex[dataVertexOffset + threadId * 3 + 1] = yToCopy;
dataVertex[dataVertexOffset + threadId * 3 + 2] = zToCopy;
}
}