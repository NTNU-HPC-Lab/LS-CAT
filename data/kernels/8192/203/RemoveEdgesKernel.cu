#include "includes.h"
__global__ void RemoveEdgesKernel(  int *connection, int *age, int maxAge, int *activityFlag, float *winningFraction, int *winningCount, float *utility, float *localError, int *neuronAge, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
if(activityFlag[threadId] == 1)
{
neuronAge[threadId] = neuronAge[threadId] + 1;

// TO DO : GET RID OF  IFs & ELSEs
int activeConnections = 0;
int connId;
for(int c = 0; c < maxCells; c++)
{
connId = threadId * maxCells + c;
if(connection[connId] == 1)
{
if(age[connId] <= maxAge)
{
activeConnections++;
}
else
{
connection[connId] = 0;
age[connId] = 0;
}
}
}
if(activeConnections == 0)
{
activityFlag[threadId] = 0;
localError[threadId] = 0.00f;
neuronAge[threadId] = 0;
winningFraction[threadId] = 0.00f;
winningCount[threadId] = 0;
utility[threadId] = 0.00f;
}
}
}
}