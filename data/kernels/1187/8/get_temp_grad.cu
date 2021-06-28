#include "includes.h"
__global__ void get_temp_grad (const int n, const float *gradOutput, const float *mask, float *top_grad, const int mask_index){
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
if (((int) mask[index]) == mask_index)
top_grad[index] = gradOutput[index];
}