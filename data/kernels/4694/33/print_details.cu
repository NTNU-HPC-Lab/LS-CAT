#include "includes.h"
__global__ void print_details()
{
printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y : %d, gridDim.x : %d, gridDim.y :%d \n",
blockIdx.x, blockIdx.y, blockIdx.z,blockDim.x, blockDim.y, gridDim.x, gridDim.y);
}