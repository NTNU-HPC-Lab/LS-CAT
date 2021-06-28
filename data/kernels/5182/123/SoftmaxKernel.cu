#include "includes.h"
__global__ void SoftmaxKernel( float *outputPtr, float expSum, int layerSize )
{
// i: neuron id
int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < layerSize)
{
// exp value is already present in the output array, so just divide by sum of exps (computed before kernel call)
outputPtr[i] /= expSum;
}


}