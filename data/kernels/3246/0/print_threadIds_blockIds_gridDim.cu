#include "includes.h"



__global__ void print_threadIds_blockIds_gridDim()
{
printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d,\
blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d,\
gridDim.x: %d, gridDim.y: %d, gridDim.z: %d \n",
threadIdx.x, threadIdx.y, threadIdx.z,
blockIdx.x, blockIdx.y, blockIdx.z,
gridDim.x, gridDim.y, gridDim.z);
}