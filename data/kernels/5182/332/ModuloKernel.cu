#include "includes.h"
__global__ void ModuloKernel(float* input, int divisor, float* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
if(id < size)
{
output[id] = (float)   (((int)input[id]) % divisor) ;
}
}