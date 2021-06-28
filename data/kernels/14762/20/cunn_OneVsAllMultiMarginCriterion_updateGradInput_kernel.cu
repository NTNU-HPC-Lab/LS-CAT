#include "includes.h"
__global__ void cunn_OneVsAllMultiMarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int nframe, int dim, int sizeaverage, float *positiveWeight)
{
// __shared__ float buffer[MULTIMARGIN_THREADS];
int k = blockIdx.x;
float *input_k = input + k*dim;
float *gradInput_k = gradInput + k*dim;
int target_k = ((int)target[k])-1;
float g = (sizeaverage ? 1./((float)dim) : 1.);

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

//  buffer[threadIdx.x] = 0;
for (int i=i_start; i<i_end; i+=i_step)
{
float y = (i==target_k) ? 1.0 : -1.0;
float z = 1 - input_k[i]*y;

if(z > 0)
{
float weight = (i==target_k) ? positiveWeight[i] : 1.0;
float h =  -y*g*weight;
gradInput_k[i] = h;
}
else
gradInput_k[i] = 0;
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