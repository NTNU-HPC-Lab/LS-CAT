#include "includes.h"

//device functions
__device__ int getGlobalIdx_3D_2D()
{
int blockId = blockIdx.x
+ blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;

int threadId = blockId * (blockDim.x * blockDim.y)
+ (threadIdx.y * blockDim.x)
+ threadIdx.x;

return threadId;
}
__global__ void kernel_3D_2D()
{
printf("Local thread IDs: (%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, getGlobalIdx_3D_2D());
}