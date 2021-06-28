#include "includes.h"
__global__ void kern_MinBuffers(float* b1, float* b2, int size)
{
int idx = CUDASTDOFFSET;
float value1 = b1[idx];
float value2 = b2[idx];
float minVal =  (value1 < value2) ? value1 : value2;
if( idx < size )
{
b1[idx] = minVal;
}
}