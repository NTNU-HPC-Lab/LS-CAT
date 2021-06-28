#include "includes.h"
__global__ void cunnx_BlockSparse_accGradParameters_kernel( float *gradWeight, float* gradBias, float *gradOutput, float *input, float *inputIndice, float *outputIndice, int inputSize, int outputSize, int nInputBlock, int nOutputBlock, int inputWindowSize, int outputWindowSize, float scale)
{
__shared__ float buffer[BLOCKSPARSE_THREADS];
__shared__ float gradOutputBuffer[BLOCKSPARSE_MAXOUTPUTBLOCKSIZE];
int tx = threadIdx.x;
int i_step = blockDim.x;
int k = blockIdx.x;

float *input_k = input + k*inputWindowSize*inputSize;
float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
float *inputIndice_k = inputIndice + k*inputWindowSize;
float *outputIndice_k = outputIndice + k*outputWindowSize;

// loop through blocks
for (int m=0; m<outputWindowSize; m++)
{
int outputIdx = (int)outputIndice_k[m] - 1;

float *blockGradOutput = gradOutput_k + m*outputSize;
float *blockGradBias = gradBias + outputIdx*outputSize;

for (int j=tx; j<outputSize; j+=i_step)
gradOutputBuffer[j] = blockGradOutput[j]*scale;

__syncthreads(); // needed for some reason

for (int l=0; l<inputWindowSize; l++)
{
int inputIdx = (int)inputIndice_k[l] - 1;

float *blockInput = input_k + l*inputSize;
float *blockGradWeight = gradWeight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;

// addr weights (scalar-products)
for (int i=tx; i<inputSize; i+=i_step)
{
// copy input to buffer
buffer[tx] = blockInput[i];

// multiply accumulate weights
for (int j=0; j<outputSize; j++)
atomicAdd(&(blockGradWeight[j*inputSize + i]), gradOutputBuffer[j]*buffer[tx]);
}
}

__syncthreads(); // needed for some reason

// multiply accumulate biases
for (int j=tx; j<outputSize; j+=i_step)
atomicAdd(&(blockGradBias[j]), gradOutputBuffer[j]);
}
}