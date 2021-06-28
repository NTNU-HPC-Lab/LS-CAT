#include "includes.h"
__global__ void swapVals_kernel(unsigned int * d_newArray, unsigned int * d_oldArray, unsigned int numElems)
{
unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
if (gIdx < numElems)
{
d_newArray[gIdx] = d_oldArray[gIdx];
}
}