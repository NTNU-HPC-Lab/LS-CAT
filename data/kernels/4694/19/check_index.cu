#include "includes.h"
__global__ void check_index()
{
printf(" threadidx:(%d,%d,%d) blockidx:(%d,%d,%d) blockdim:(%d,%d,%d) griddim:(%d,%d,%d) \n",
threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,
blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
}