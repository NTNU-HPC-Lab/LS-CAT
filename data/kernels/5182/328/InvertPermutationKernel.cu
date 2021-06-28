#include "includes.h"
__global__ void InvertPermutationKernel(float* input, float* output, int size)
{
int id = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if (id >= size)
return;


int temp = __float2int_rn(input[id]);

if (input == output)
__syncthreads();

output[temp] = id;
}