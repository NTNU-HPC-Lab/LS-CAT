#include "includes.h"

//device functions
__device__ int getGlobalIdx_1D_2D()
{
return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}
__global__ void kernel_1D_2D()
{
printf("Local thread IDs: (%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, getGlobalIdx_1D_2D());
}