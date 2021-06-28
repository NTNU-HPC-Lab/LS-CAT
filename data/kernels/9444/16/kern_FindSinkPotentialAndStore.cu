#include "includes.h"
__global__ void kern_FindSinkPotentialAndStore(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = workingBuffer[idx] + incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
if( idx < size )
{
workingBuffer[idx] = value;
}
}