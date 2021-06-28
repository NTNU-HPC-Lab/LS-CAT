#include "includes.h"
__global__ void MarkCentroidsKernel( float *centroidCoordinates, float *visField, int imgWidth, int imgHeight, int centroids )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;
if(threadId < centroids)
{
int x = lrintf(centroidCoordinates[threadId * 2]);
int y = lrintf(centroidCoordinates[threadId * 2 + 1]);

visField[y * imgWidth + x] = -1.00f;

}
}