#include "includes.h"
__global__ void LinearCombinationKernel(float *input1, float input1_coeff, int input1_start_index, float *input2, float input2_coeff, int input2_start_index, float *output, int output_start_index, int size)
{
int id = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if(id < size)
{
output[output_start_index + id] = input1_coeff * input1[input1_start_index + id] + input2_coeff * input2[input2_start_index + id];
}
}