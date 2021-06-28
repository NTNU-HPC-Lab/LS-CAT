#include "includes.h"
__global__ void kern_FindSourcePotentialAndStore(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = workingBuffer[idx] + sinkBuffer[idx] + divBuffer[idx] - labelBuffer[idx] * iCC;
if( idx < size )
{
workingBuffer[idx] = value;
}
}