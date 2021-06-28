#include "includes.h"
__global__ void CopyConnectionsCoordinatesKernel(   int *connectionMatrix, float *pointsCoordinates, float *vertexData, int *connectionCount, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells * maxCells)
{
if(connectionMatrix[threadId] == 1)
{
int from = threadId / maxCells;
int to = threadId % maxCells;

if(to > from)
{
//int vertexDataOffset = maxCells * 3;
int vertexDataOffset = 0;
int connIdx = atomicAdd( &connectionCount[0], 1);

vertexData[vertexDataOffset + connIdx * 6] = pointsCoordinates[from * 3];
vertexData[vertexDataOffset + connIdx * 6 + 1] = pointsCoordinates[from * 3 + 1];
vertexData[vertexDataOffset + connIdx * 6 + 2] = pointsCoordinates[from * 3 + 2];

vertexData[vertexDataOffset + connIdx * 6 + 3] = pointsCoordinates[to * 3];
vertexData[vertexDataOffset + connIdx * 6 + 4] = pointsCoordinates[to * 3 + 1];
vertexData[vertexDataOffset + connIdx * 6 + 5] = pointsCoordinates[to * 3 + 2];
}

}

}
}