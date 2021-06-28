#include "includes.h"
__global__ void cunnx_LazyKBest_updateGradInput_kernel( float *gradInput, const float *indice, const float *gradOutput, int inputSize, int outputSize)
{
int tx = threadIdx.x;
int step = blockDim.x;
int k = blockIdx.x;

float *gradInput_k = gradInput + k*inputSize;
const float *gradOutput_k = gradOutput + k*outputSize;
const float *indice_k = indice + k*outputSize;

for (int i=tx; i<outputSize; i+=step)
gradInput_k[(int)(indice_k[i] - 1)] = gradOutput_k[i];
}