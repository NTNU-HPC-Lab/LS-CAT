#include "includes.h"
__global__ void kern_UpdateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size)
{
int idx = CUDASTDOFFSET;
float value = labelBuffer[idx] + CC*(incBuffer[idx] - divBuffer[idx] - sinkBuffer[idx]);
value = saturate(value);
if( idx < size )
{
labelBuffer[idx] = value;
}
}