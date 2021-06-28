#include "includes.h"

//device functions
__device__ int getGlobalIdx_1D_3D()
{
return blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}
__global__ void kernel_1D_3D()
{
printf("Local thread IDs: (%i,%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, threadIdx.z, getGlobalIdx_1D_3D());
}