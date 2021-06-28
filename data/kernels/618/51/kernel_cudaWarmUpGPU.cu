#include "includes.h"
__global__ void kernel_cudaWarmUpGPU()
{
int ind=blockIdx.x*blockDim.x+threadIdx.x;
ind = ind + 1;
}