#include "includes.h"
__global__ void oneOfNSelection(float *buffer, int* index, int size, float value)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x						//blocks preceeding current block
+ threadIdx.x;

if (threadId < size && threadId != index[0])
{
buffer[threadId] = 0;

}
else if (threadId < size && threadId == index[0]){
buffer[threadId] = value;
}
}