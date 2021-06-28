#include "includes.h"
__global__ void kern_CalcGradStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = stepSize*(sinkBuffer[idx] + divBuffer[idx] - incBuffer[idx] - labelBuffer[idx] * iCC);
if( idx < size )
{
divBuffer[idx] = value;
}
}