#include "includes.h"
__global__ void cunnx_BlockSparse_updateOutput_kernel( float *output, const float *input, const float *outputIndice, const float *outputScale, const float *bias, int outputSize, int nOutputBlock, int inputWindowSize, int outputWindowSize)
{
__shared__ float buffer[BLOCKSPARSE_THREADS];
int tx = threadIdx.x;
int i_step = blockDim.x;
int k = blockIdx.x;

float *output_k = output + k*outputWindowSize*outputSize;
const float *input_k = input + k*inputWindowSize*outputWindowSize*outputSize;
const float *outputIndice_k = outputIndice + k*outputWindowSize;
const float *outputScale_k = outputScale + k*outputWindowSize;

for (int m=0; m<outputWindowSize; m++)
{
int outputIdx = (int)outputIndice_k[m] - 1;
float outputScale = outputScale_k[m];

for (int j=tx; j<outputSize; j+=i_step)
{
buffer[tx] = bias[outputIdx*outputSize + j];

for (int l=0; l<inputWindowSize; l++)
buffer[tx] += input_k[l*outputWindowSize*outputSize + m*outputSize + j];

output_k[m*outputSize + j] = outputScale*buffer[tx];
}
}
}