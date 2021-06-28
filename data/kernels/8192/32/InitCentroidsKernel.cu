#include "includes.h"
__global__ void InitCentroidsKernel(  float *centroidCoordinates, float *randomNumbers, float minX, float maxX, float minY, float maxY, int centroids  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < centroids)
{
centroidCoordinates[threadId *2] = (maxX - minX) * randomNumbers[threadId * 2] + minX;
centroidCoordinates[threadId * 2 + 1] = (maxY - minY) * randomNumbers[threadId * 2 + 1] + minY;
}
}