#include "includes.h"
__global__ void kern_PushUpSourceFlows(float* psink, float* sink, float* source, float* div, float* label, float w, float iCC, int size)
{
int idx = CUDASTDOFFSET;
float value = psink[idx] + w*(sink[idx] - source[idx] + div[idx] - label[idx] * iCC);
if( idx < size )
{
psink[idx] = value;
}
}