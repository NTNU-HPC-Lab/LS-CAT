#include "includes.h"
__global__ void predicate(int *d_array, int d_numberOfElements,int *d_predicateArray)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index <d_numberOfElements)
{
if(d_array[index]%32== 0)
{
d_predicateArray[index] =1;
}
else
{
d_predicateArray[index]  = 0;
}
}
}