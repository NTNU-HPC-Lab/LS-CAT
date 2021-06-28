#include "includes.h"
__global__ void AddAndRefreshConnectionKernel(  int node1, int node2, int *activityFlag, int *connection, int *age, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
activityFlag[node1] = 1;
activityFlag[node2] = 1;

connection[node1 * maxCells + node2] = 1;
age[node1 * maxCells + node2] = 0;
connection[node2 * maxCells + node1] = 1;
age[node2 * maxCells + node1] = 0;
}
}