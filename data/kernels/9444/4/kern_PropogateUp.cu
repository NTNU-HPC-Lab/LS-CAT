#include "includes.h"
__global__ void kern_PropogateUp(float* working, int span, int imageSize)
{
int idx = CUDASTDOFFSET;
float inputValue1 = working[idx];
float inputValue2 = working[idx+span];
float outputVal = (inputValue1 > inputValue2) ? inputValue1: inputValue2;
if(idx+span < imageSize)
{
working[idx] = outputVal;
}
}