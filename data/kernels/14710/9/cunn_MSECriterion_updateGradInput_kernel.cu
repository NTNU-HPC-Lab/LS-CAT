#include "includes.h"
__global__ void cunn_MSECriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, float norm, int nframe, int dim)
{
int k = blockIdx.x;
float *gradInput_k = gradInput + k*dim;
float *input_k = input + k*dim;
float *target_k = target + k*dim;

int i_start = threadIdx.x;
int i_end = dim;
int i_step = blockDim.x;

// gradInput
for (int i=i_start; i<i_end; i+=i_step)
gradInput_k[i] = norm*(input_k[i] - target_k[i]);
}