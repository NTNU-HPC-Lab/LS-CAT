#include "includes.h"
__global__ void kern_Copy2Buffers(float* fIn, float* fOut1, float* fOut2, int size)
{
int idx = CUDASTDOFFSET;
float value = fIn[idx];
if( idx < size )
{
fOut1[idx] = value;
}
if( idx < size )
{
fOut2[idx] = value;
}
}