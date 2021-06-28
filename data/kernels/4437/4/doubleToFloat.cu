#include "includes.h"
__global__ void doubleToFloat(double* input, float* output, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numElements)
{
output[i] = (float)input[i];
}
}