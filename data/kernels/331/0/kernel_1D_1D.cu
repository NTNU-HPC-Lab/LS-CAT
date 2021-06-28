#include "includes.h"

//device functions
__device__ int getGlobalIdx_1D_1D()
{
return blockIdx.x *blockDim.x + threadIdx.x;
}
__global__ void kernel_1D_1D()
{
printf("Local thread ID: %i   Global thread ID: %i\n", threadIdx.x, getGlobalIdx_1D_1D());
}