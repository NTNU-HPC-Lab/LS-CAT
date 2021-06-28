#include "includes.h"
__global__ void GaussianMinMaxField(float* input, int inputCount, float* mins, float* maxes)
{
int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < inputCount)
{
mins[i] = fminf(mins[i], input[i]);
maxes[i] = fmaxf(maxes[i], input[i]);
}
}