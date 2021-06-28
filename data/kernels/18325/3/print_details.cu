#include "includes.h"
__global__ void print_details ()
{
printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx: %d, \
blockDim.x: %d, blockDim.y: %d, blockDim: %d, \
gridDim.x: %d, gridDim.y: %d, girdDim: %d\n", \
blockIdx.x, blockIdx.y, blockIdx.z, \
blockDim.x, blockDim.y, blockDim.z, \
gridDim.x, gridDim.y, gridDim.z);
}