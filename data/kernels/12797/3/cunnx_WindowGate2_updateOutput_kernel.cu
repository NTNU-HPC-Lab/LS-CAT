#include "includes.h"
__global__ void cunnx_WindowGate2_updateOutput_kernel( float *output, float *centroids, float *normalizedCentroids, float *inputIndice, float *outputIndice, const float *input, const float *noise, int inputSize, int outputSize, int inputWindowSize, int outputWindowSize, int windowStride, int train)
{
__shared__ float buffer[WINDOWGATE2_THREADS+1];
unsigned int tx = threadIdx.x;
unsigned int k = blockIdx.x;
const float *input_k = input + inputSize*k;
float *output_k = output + outputWindowSize*k;

// get coordinate of centoid
buffer[tx] = 0;
for (unsigned int i=tx; i<inputSize; i+=blockDim.x)
buffer[tx] += input_k[i]*(float)(i+1);

// add (reduce)
for (unsigned int stride = WINDOWGATE2_THREADS >> 1; stride > 0; stride >>= 1)
{
__syncthreads();
if (tx < stride)
buffer[tx] += buffer[tx+stride];
}

if (tx == 0)
{
float centroid = buffer[0];

// make centroid a number between 0 and 1
centroid /= (float)(inputSize);

normalizedCentroids[k] = centroid;
if ( train )
{
centroid += noise[k];
centroid = fminf(fmaxf(0,centroid),1);
}
// align centroid to output
centroid *= (float)(outputSize);

float inputIdx = centroid/(float)(inputSize) - 0.5*(float)inputWindowSize;
float outputIdx = centroid - 0.5*(float)outputWindowSize;

// clip indices
inputIdx = fminf(inputIdx, inputSize-inputWindowSize+1);
inputIdx = fmaxf(inputIdx, 1);
outputIdx = fminf(outputIdx, outputSize-outputWindowSize+1);
outputIdx = fmaxf(outputIdx, 1);

inputIdx = ceilf(inputIdx);
outputIdx = ceilf(outputIdx);
// align centroid to outputWindow
centroid -= (outputIdx-1);

inputIndice[k] = (int)inputIdx;
outputIndice[k] = (int)outputIdx;
centroids[k] = centroid;

buffer[WINDOWGATE2_THREADS] = inputIdx;
}

__syncthreads();

float inputIdx = buffer[WINDOWGATE2_THREADS];
const float *inputWindow = input_k + (int)inputIdx;

for (int i=tx; i<outputWindowSize; i+=blockDim.x)
{
output_k[i] = inputWindow[(int)floorf(((float)i)/windowStride)];
}
}