#include "includes.h"
__global__ void cunn_SoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int nframe, int dim, int stride)
{
__shared__ float buffer[SOFTMAX_THREADS];
float *gradInput_k = gradInput + blockIdx.x*dim*stride + blockIdx.y;
float *output_k = output + blockIdx.x*dim*stride + blockIdx.y;
float *gradOutput_k = gradOutput + blockIdx.x*dim*stride + blockIdx.y;

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

// sum?
buffer[threadIdx.x] = 0;
for (int i=i_start; i<i_end; i+=i_step)
buffer[threadIdx.x] += gradOutput_k[i*stride] * output_k[i*stride];

__syncthreads();

// reduce
if (threadIdx.x == 0)
{
float sum_k = 0;
for (int i=0; i<blockDim.x; i++)
sum_k += buffer[i];
buffer[0] = sum_k;
}

__syncthreads();

float sum_k = buffer[0];
for (int i=i_start; i<i_end; i+=i_step)
gradInput_k[i*stride] = output_k[i*stride] * (gradOutput_k[i*stride] - sum_k);
}