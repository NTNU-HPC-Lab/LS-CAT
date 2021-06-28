#include "includes.h"
__global__ void vectFill( int * data1, int * data2, int * restult, unsigned long sizeOfArray )
{
unsigned long i = blockDim.x * blockIdx.x + threadIdx.x;
if( i < sizeOfArray )
{
restult[ i ] = data1[i] + data2[i];
}
}