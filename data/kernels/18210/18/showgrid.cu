#include "includes.h"
__global__ void showgrid(){
printf("thread: %d, %d %d\nblock Idxs: %d, %d %d\nblock Dims: %d, %d %d\ngrid: %d, %d %d\n\n\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}