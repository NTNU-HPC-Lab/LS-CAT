#include "includes.h"
__global__ void kern_Lbl(float* lbl, float* flo, float* cap, const int size)
{
int idx = CUDASTDOFFSET;
float value1 = cap[idx];
float value2 = flo[idx];
float minVal =  (value2 == value1) ? 1.0f : 0.0f;
if( idx < size )
{
lbl[idx] = minVal;
}
}