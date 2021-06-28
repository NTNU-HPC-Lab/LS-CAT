#include "includes.h"
__global__ void kern_FindLeafSinkPotential(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
if( idx < size )
{
sinkBuffer[idx] = value;
}
}