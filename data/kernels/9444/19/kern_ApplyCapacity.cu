#include "includes.h"
__global__ void kern_ApplyCapacity(float* sinkBuffer, float* capBuffer, int size)
{
int idx = CUDASTDOFFSET;
float value = sinkBuffer[idx];
float cap = capBuffer[idx];
value = (value < 0.0f) ? 0.0f: value;
value = (value > cap) ? cap: value;
if( idx < size )
{
sinkBuffer[idx] = value;
}
}