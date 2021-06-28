#include "includes.h"
__global__ void cunn_OneVsAllMultiMarginCriterion_updateOutput_kernel(float *output, float *input, float *target, int nframe, int dim, int sizeaverage, float *positiveWeight)
{
__shared__ float buffer[MULTIMARGIN_THREADS];
int k = blockIdx.x;
float *input_k = input + k*dim;
float *output_k = output + k;
int target_k = ((int)target[k])-1;

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

buffer[threadIdx.x] = 0;
for(int i = i_start; i < i_end; i += i_step)
{

float y = (i==target_k) ? 1.0 : -1.0;
float z = 1 - input_k[i]*y;
if(z > 0){
float weight = (i==target_k) ? positiveWeight[i] : 1.0;
buffer[threadIdx.x] += z*weight;
}
}
__syncthreads();

// reduce
if (threadIdx.x == 0)
{
float sum = 0;
for (int i=0; i<blockDim.x; i++)
sum += buffer[i];

if(sizeaverage)
*output_k = sum/dim;
else
*output_k = sum;
}
}