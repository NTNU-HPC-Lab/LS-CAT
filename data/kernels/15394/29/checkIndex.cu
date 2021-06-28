#include "includes.h"
__global__ void checkIndex(void)
{
printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}