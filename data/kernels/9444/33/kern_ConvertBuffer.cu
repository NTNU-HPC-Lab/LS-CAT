#include "includes.h"
__global__ void kern_ConvertBuffer(short* agreement, float* output, int size )
{
int idx = CUDASTDOFFSET;
float locAgreement = (float) agreement[idx];
if( idx < size )
{
output[idx] = locAgreement;
}
}