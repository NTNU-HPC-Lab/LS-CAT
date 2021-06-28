#include "includes.h"
__global__ void PolynomialFunctionKernel_Double(float a3, float a2, float a1, float a0, double* input, double* output, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (id < size)
{
double x = input[id];
output[id] = a3 * x * x * x + a2 * x * x + a1 * x + a0;
}
}