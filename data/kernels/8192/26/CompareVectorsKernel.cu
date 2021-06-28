#include "includes.h"
__global__ void CompareVectorsKernel(float* inputOne, float* inputTwo, float* output)
{
int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (inputOne[id] != inputTwo[id])
output[0] = 1;
}