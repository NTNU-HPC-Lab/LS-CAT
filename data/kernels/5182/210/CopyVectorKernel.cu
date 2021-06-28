#include "includes.h"
__global__ void CopyVectorKernel(  float *from, int fromOffset, float *to, int toOffset, int vectorSize  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < vectorSize)
{
to[threadId + toOffset] = from[threadId + fromOffset];
}

}