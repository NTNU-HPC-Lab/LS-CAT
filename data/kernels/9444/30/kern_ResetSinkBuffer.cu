#include "includes.h"
__global__ void kern_ResetSinkBuffer(float* sink, float* source, float* div, float* label, float ik, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = (1.0f-ik)*sink[idx] + ik*(source[idx] - div[idx] + label[idx] * iCC);
if( idx < size )
{
sink[idx] = value;
}
}