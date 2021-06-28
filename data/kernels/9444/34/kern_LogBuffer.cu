#include "includes.h"
__global__ void kern_LogBuffer(float* agreement, float* output, float maxOut, int size, short max)
{
int idx = CUDASTDOFFSET;
float locAgreement = (float) agreement[idx];
float logValue = (locAgreement > 0.0f) ? log((float)max)-log(locAgreement): maxOut;
logValue = (logValue > 0.0f) ? logValue : 0.0f;
logValue = (logValue < maxOut) ? logValue: maxOut;
if( idx < size )
{
output[idx] = logValue;
}
}