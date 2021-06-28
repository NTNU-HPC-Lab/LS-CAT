#include "includes.h"
__global__ void hillisSteeleScanDevice(int *d_array , int numberOfElements, int *d_tmpArray,int moveIndex)
{
int index = threadIdx.x + blockDim.x * blockIdx.x;
if(index > numberOfElements)
{
return;
}
d_tmpArray[index] = d_array[index];
if(index - moveIndex >=0)
{

d_tmpArray[index] = d_tmpArray[index] +d_array[index - moveIndex];
}
}