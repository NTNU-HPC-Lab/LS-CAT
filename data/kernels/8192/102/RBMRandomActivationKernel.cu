#include "includes.h"
__device__ float activateRandomly(float probability, float random)
{
return random < probability;
}
__global__ void RBMRandomActivationKernel( float					*outputPtr, float					*randomPtr, int						size )
{

int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < size)
{
outputPtr[i] = activateRandomly(outputPtr[i], randomPtr[i]);
}
}