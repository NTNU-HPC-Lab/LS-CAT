#include "includes.h"
__global__ void kern_ProbBuffer(float* agreement, float* output, int size, short max)
{
int idx = CUDASTDOFFSET;
float locAgreement = agreement[idx];
float probValue = (float) locAgreement / (float) max;
probValue = (probValue < 1.0f) ? probValue: 1.0f;
if( idx < size )
{
output[idx] = probValue;
}
}