#include "includes.h"
__global__ void CUDAkernel_accumulate( float* buffer, int addSize, int size )
{
int index = CUDASTDOFFSET;
float a = buffer[index];
float b = buffer[index+addSize];
if( index+addSize < size )
{
buffer[index] = a+b;
}
}