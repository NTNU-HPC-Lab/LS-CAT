#include "includes.h"
__global__ void DropoutMaskKernel( float *dropoutMaskPtr, float dropout, int inputSize )
{
int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < inputSize)
{
dropoutMaskPtr[i] = dropout > dropoutMaskPtr[i];
/*if (dropoutMaskPtr[i] > dropout)
dropoutMaskPtr[i] = 0.0f;
else
dropoutMaskPtr[i] = 1.0f;*/
}
}