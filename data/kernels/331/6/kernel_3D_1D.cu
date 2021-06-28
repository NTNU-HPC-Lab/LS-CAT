#include "includes.h"

//device functions
__device__ int getGlobalIdx_3D_1D()
{
int blockId = blockIdx.x
+ blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;

int threadId = blockId * blockDim.x + threadIdx.x;

return threadId;
}
__global__ void kernel_3D_1D()
{
printf("Local thread ID: %i   Global thread ID: %i\n", threadIdx.x, getGlobalIdx_3D_1D());
}