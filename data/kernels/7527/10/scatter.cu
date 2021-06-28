#include "includes.h"
__global__ void scatter(int *d_array , int *d_predicateArray, int *d_scanArray,int *d_compactedArray, int d_numberOfElements)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < d_numberOfElements)
{
if(d_predicateArray[index]==1)
{
d_compactedArray[d_scanArray[index]-1] = d_array[index];

}
}
}