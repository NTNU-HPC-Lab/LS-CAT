#include "includes.h"
__global__ void CenterOfGravityKernel(  float *pointsCoordinates, float *centerOfGravity, int *activityFlag, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
float xSum = 0.00f, ySum = 0.00f, zSum = 0.00f;
int livingCells = 0;
for(int c = 0; c < maxCells; c++)
{
if(activityFlag[c] == 1)
{
xSum += pointsCoordinates[c * 3];
ySum += pointsCoordinates[c * 3 + 1];
zSum += pointsCoordinates[c * 3 + 2];

livingCells++;
}
}
centerOfGravity[0] = xSum / (float)livingCells;
centerOfGravity[1] = ySum / (float)livingCells;
centerOfGravity[2] = zSum / (float)livingCells;
}
}