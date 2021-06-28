#include "includes.h"
__global__ void hillisSteeleScanDevice(int *d_predicateArray , int d_numberOfElements ,int *d_tmpArray,int d_offset)
{
int index = blockIdx.x * blockDim.x +  threadIdx.x;
if(index < d_numberOfElements)
{
d_tmpArray[index] = d_predicateArray[index];
if(index - d_offset >= 0)
{

d_tmpArray[index] = d_predicateArray[index] + d_predicateArray[index-d_offset];
}
}
}