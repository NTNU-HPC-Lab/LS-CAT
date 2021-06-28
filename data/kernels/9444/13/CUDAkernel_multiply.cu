#include "includes.h"
__global__ void CUDAkernel_multiply( float* sourceA, float* sourceB, float* destination, int size )
{
int index = CUDASTDOFFSET;
float a = sourceA[index];
float b = sourceB[index];
if( index < size )
{
destination[index] = a * b;
}
}