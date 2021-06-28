#include "includes.h"
__global__ void predicateDevice(int *d_array , int *d_predicateArrry , int d_numberOfElements,int bit,int bitset)
{
int index = threadIdx.x + blockIdx.x*blockDim.x;
if(index < d_numberOfElements)
{
if(bitset == 0)
{
if((d_array[index] & bit) == 0)
{
d_predicateArrry[index] = 1;
}
else
{
d_predicateArrry[index] = 0;
}
}
else
{
if((d_array[index] & bit) != 0)
{
d_predicateArrry[index] = 1;
}
else
{
d_predicateArrry[index] = 0;
}
}
}
}