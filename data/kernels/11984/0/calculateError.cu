#include "includes.h"
__global__ void calculateError(float *aFourth, float *err, int expectedOutput)
{
int i = threadIdx.x;
err[i] = aFourth[i] - (i + 1 == expectedOutput);
}