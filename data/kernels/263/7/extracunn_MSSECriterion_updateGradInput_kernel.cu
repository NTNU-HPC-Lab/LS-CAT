#include "includes.h"
__global__ void extracunn_MSSECriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, float norm, int nframe, int dim)
{
int k = blockIdx.x;
float *gradInput_k = gradInput + k*dim;
float *input_k = input + k*dim;
float *target_k = target + k*dim;

__shared__ float buffer[MSSECRITERION_THREADS];

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;
float sum = 0.0;
// msse
buffer[threadIdx.x] = 0;
for (int i=i_start; i<i_end; i+=i_step)
{
float z = input_k[i] - target_k[i];
buffer[threadIdx.x] += z;
}
__syncthreads();


//reduce
if (threadIdx.x == 0)
{
sum = 0;
for (int i=0; i<blockDim.x; i++)
{
sum += buffer[i];
}
}

// gradInput
for (int i=i_start; i<i_end; i+=i_step)
gradInput_k[i] = norm*sum;
}