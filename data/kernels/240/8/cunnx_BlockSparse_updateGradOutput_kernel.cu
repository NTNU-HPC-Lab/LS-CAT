#include "includes.h"
__global__ void cunnx_BlockSparse_updateGradOutput_kernel( float *_gradOutput, float* gradOutputScale, const float *gradOutput, const float *output, const float *outputScale, int outputWindowSize, int outputSize)
{
__shared__ float buffer[BLOCKSPARSE_THREADS];
int tx = threadIdx.x;
int i_step = blockDim.x;
int k = blockIdx.x;

float *_gradOutput_k = _gradOutput + k*outputWindowSize*outputSize;
float *gradOutputScale_k = gradOutputScale + k*outputWindowSize;
const float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
const float *output_k = output + k*outputWindowSize*outputSize;
const float *outputScale_k = outputScale + k*outputWindowSize;


// get gradients for outputScale (to be backwarded to a Gater)
for (int m=0; m<outputWindowSize; m++)
{
float outputScale = outputScale_k[m];

float *_blockGradOutput = _gradOutput_k + m*outputSize;
const float *blockGradOutput = gradOutput_k + m*outputSize;
const float *blockOutput = output_k + m*outputSize;

buffer[tx] = 0;

for (int j=tx; j<outputSize; j+=i_step)
{
const float grad = blockGradOutput[j];
buffer[tx] += blockOutput[j]*grad;
_blockGradOutput[j] = grad*outputScale;
}

// add (reduce)
for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
{
__syncthreads();
if (tx < stride)
buffer[tx] += buffer[tx+stride];
}

if (tx == 0)
gradOutputScale_k[m] = buffer[0]/(outputScale+0.00000001);
}
}