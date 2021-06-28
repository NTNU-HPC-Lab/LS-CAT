#include "includes.h"
__global__ void cunn_OneVsAllNLLCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int nframe, int dim, int sizeaverage, float *positiveWeight)
{
// __shared__ float buffer[NLL_THREADS];
int k = blockIdx.x;
float *input_k = input + k*dim;
float *gradInput_k = gradInput + k*dim;
int target_k = ((int)target[k])-1;
float g = (sizeaverage ? 1./((float)nframe) : 1.);

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

//  buffer[threadIdx.x] = 0;
for (int i=i_start; i<i_end; i+=i_step)
{
float z = (i==target_k) ? -g*positiveWeight[i]/input_k[i] : g/(1 - input_k[i]);
gradInput_k[i] = z;
}

__syncthreads();

// reduce
//if (threadIdx.x == 0)
//{
// float gradInput_target_k = 0;
//for (int i=0; i<blockDim.x; i++)
// gradInput_target_k += buffer[i];
//gradInput_k[target_k] = gradInput_target_k;
//}
}