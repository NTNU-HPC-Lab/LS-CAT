#include "includes.h"
__global__ void extracunn_MSSECriterion_updateOutput_kernel(float* output, float *input, float *target, int nframe, int dim)
{
__shared__ float buffer[MSSECRITERION_THREADS];
int k = blockIdx.x;
float *input_k = input + k*dim;
float *target_k = target + k*dim;

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

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
*output = 0;
for (int i=0; i<blockDim.x; i++)
{
*output += buffer[i];
}
*output *= (*output);
*output /= (-2*dim*dim);
}
}