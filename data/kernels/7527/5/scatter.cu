#include "includes.h"
__global__ void scatter(int *d_array , int *d_scanArray , int *d_predicateArrry,int * d_scatteredArray ,int d_numberOfElements,int offset)
{
int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < d_numberOfElements)
{
if(d_predicateArrry[index] == 1)
{
d_scatteredArray[d_scanArray[index] - 1 +offset ] = d_array[index];

}
}
}