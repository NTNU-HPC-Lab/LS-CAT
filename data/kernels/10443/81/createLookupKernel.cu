#include "includes.h"
__global__ void createLookupKernel(const int* inds, int total, int* output)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if (idx < total)
output[inds[idx]] = idx;
}