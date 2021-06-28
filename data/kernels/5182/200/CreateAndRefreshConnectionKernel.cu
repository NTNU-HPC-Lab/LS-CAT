#include "includes.h"
__global__ void CreateAndRefreshConnectionKernel(  int s1, int s2, int *connection, int *age, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
connection[s1 * maxCells + s2] = 1;
age[s1 * maxCells + s2] = 0;
connection[s2 * maxCells + s1] = 1;
age[s2 * maxCells + s1] = 0;
}
}