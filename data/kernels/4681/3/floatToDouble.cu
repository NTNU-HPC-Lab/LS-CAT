#include "includes.h"
__global__ void floatToDouble(float* input, double* output, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numElements)
{
output[i] = (double)input[i];
}
}