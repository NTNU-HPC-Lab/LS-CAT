#include "includes.h"
__global__ void GaussianSamplePrior(float* input, int inputCount, float* mins, float* maxes, float* randomUniform)
{
int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < inputCount)
{
float diff = maxes[i] - mins[i];
input[i] = randomUniform[i] * diff + mins[i];
}
}