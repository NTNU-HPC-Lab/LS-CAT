#include "includes.h"
__global__ void arraySet_kernel(unsigned int* d_vals, unsigned int value, size_t num_vals)
{
//	tIdx = threadIdx.x;
unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
if (gIdx < num_vals) d_vals[gIdx] = value;
}