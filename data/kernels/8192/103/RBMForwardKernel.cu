#include "includes.h"
__device__ float activationProbability(float x, float sigma)
{
return 1.0 / (1.0 + expf(-sigma * x));
}
__global__ void RBMForwardKernel( float					*inputPtr, float					*outputPtr, float					*weightPtr, float					*biasPtr, float					sigma, int						prevLayerSize, int						thisLayerSize, bool					useDropoutMask, bool					useDropout, float					dropoutRate, float					*dropoutMask )
{
// i: prev. layer neuron id
// j: current layer neuron id
int i;
int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (j < thisLayerSize)
{
// dropout this neuron
if (useDropout && useDropoutMask && !dropoutMask[j])
{
outputPtr[j] = 0;
}
else
{
float sum = 0.0;
int index = j;
for (i = 0; i < prevLayerSize; i++) {
sum += weightPtr[index] * inputPtr[i];
index += thisLayerSize;
}
// add bias
sum += biasPtr[j];

float result = activationProbability(sum, sigma);

// only used for reconstruction forward calls
if (useDropout && !useDropoutMask && dropoutRate < 1)
result /= dropoutRate;


// set output value
outputPtr[j] = result;
}


}
}