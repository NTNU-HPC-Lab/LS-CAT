#include "includes.h"
__global__ void NewNodeConnectionKernel(  int f, int q, int r, int *activityFlag, int *connection, int *age, float *localError, float alfa, int maxCells, float errorFraction  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
activityFlag[r] = 1;

connection[q * maxCells + f] = 0;
age[q * maxCells + f] = 0;
connection[f * maxCells + q] = 0;
age[f * maxCells + q] = 0;
connection[q * maxCells + r] = 1;
age[q * maxCells + r] = 0;
connection[r * maxCells + q] = 1;
age[r * maxCells + q] = 0;
connection[f * maxCells + r] = 1;
age[f * maxCells + r] = 0;
connection[r * maxCells + f] = 1;
age[r * maxCells + f] = 0;

localError[q] -= alfa * localError[q];
localError[f] -= alfa * localError[f];

localError[r] = errorFraction * (localError[q] + localError[f]);
}
}