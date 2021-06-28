#include "includes.h"
__global__ void kern_DivideBuffers(float* dst, float* src, const int size)
{
int idx = CUDASTDOFFSET;
float value1 = src[idx];
float value2 = dst[idx];
float minVal =  value2 / value1;
if( idx < size )
{
dst[idx] = minVal;
}
}