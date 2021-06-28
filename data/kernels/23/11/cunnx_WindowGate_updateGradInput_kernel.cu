#include "includes.h"
__global__ void cunnx_WindowGate_updateGradInput_kernel( float *gradInput, float *error, float* targetCentroids, const float *centroids,const float *input, const float *outputIndice, const float* output, const float* gradOutput, int inputSize, int outputSize, int outputWindowSize, float c, float d, float e, float lr)
{
__shared__ float buffer[WINDOWGATE_THREADS+1];
unsigned int tx = threadIdx.x;
unsigned int k = blockIdx.x;
const float *gradOutput_k = gradOutput + outputWindowSize*k;
const float *output_k = output + outputWindowSize*k;
const float *input_k = input + inputSize*k;
float *gradInput_k = gradInput + inputSize*k;
float centroid = centroids[k];

// get gradient of centroid
buffer[tx] = 0;
for (unsigned int i=tx; i<outputWindowSize; i+=blockDim.x)
{
buffer[tx] += gradOutput_k[i]*output_k[i]*((float)(i+1) - centroid);
}

// add (reduce)
for (unsigned int stride = WINDOWGATE_THREADS >> 1; stride > 0; stride >>= 1)
{
__syncthreads();
if (tx < stride)
buffer[tx] += buffer[tx+stride];
}

if (tx == 0)
{
int outputIdx = outputIndice[k];
float gradCentroid = buffer[0]*c;
centroid -= (lr*gradCentroid);
centroid += outputIdx-1;
centroid /= (float)(outputSize);
targetCentroids[k] = centroid;
buffer[WINDOWGATE_THREADS] = centroid*(float)(inputSize);
}

__syncthreads();
float targetCentroid = buffer[WINDOWGATE_THREADS];

buffer[tx] = 0;
// target is a gaussian blur
for (int i=tx; i<inputSize; i+=blockDim.x)
{
float target = (float)(i+1)-targetCentroid;
target = d*expf(target*target*e);
float input = input_k[i];
// dot product of logProbInput and probTarget (NLL)
buffer[tx] -= logf(input + 0.0000001)*target;
// grad input w.r.t. NLL
gradInput_k[i] = -target/(input + 0.0000001);
}

// add (reduce)
for (unsigned int stride = WINDOWGATE_THREADS >> 1; stride > 0; stride >>= 1)
{
__syncthreads();
if (tx < stride)
buffer[tx] += buffer[tx+stride];
}

if (tx == 0)
error[k] = buffer[tx];
}