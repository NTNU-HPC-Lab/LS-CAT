#include "includes.h"
__global__ void set_kernel(int* dst, int const value, int const count)
{
int const index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= count)
return;

dst[index] = value;
}