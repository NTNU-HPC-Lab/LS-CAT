#include "includes.h"
__global__ void IncrementConnectionAgeKernel(  int cell, int *connection, int *age, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
if(connection[cell * maxCells + threadId] == 1)
{
age[cell * maxCells + threadId] += 1;
age[threadId * maxCells + cell] += 1;
}

}
}