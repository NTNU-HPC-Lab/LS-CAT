#include "includes.h"
__global__ void LinearFunctionKernelDouble(double a1, double a0, double* input, double* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if(id < size)
{
double x = input[id];
output[id] =  a1 * x + a0;
}
}