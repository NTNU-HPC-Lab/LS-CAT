#include "includes.h"
__global__ void CopyInputToVisFieldKernel(  float *input, float *visField,  int inputSize )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
visField[threadId] = input[threadId];
}
}