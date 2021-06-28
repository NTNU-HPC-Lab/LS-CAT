#include "includes.h"
__global__ void _cuda_add_scalar(int *in, int scalar, int n)
{
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
while(globalIdx < n)
{
in[globalIdx] = in[globalIdx] + scalar;
globalIdx += blockDim.x * gridDim.x;
}
}