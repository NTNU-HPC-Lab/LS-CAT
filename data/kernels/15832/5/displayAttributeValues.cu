#include "includes.h"
__global__ void displayAttributeValues() {

printf("\nthreadIdx.x : %d,  threadIdx.y : %d,  threadIdx.z : %d,"
"  blockIdx.x : %d,  blockIdx.y : %d,  blockIdx.z : %d,"
"  blockDim.x : %d,  blockDim.y : %d,  blockDim.z : %d,"
"  gridDim.x : %d,  gridDim.y : %d,  gridDim.z : %d\n",
threadIdx.x,threadIdx.y,threadIdx.z,
blockIdx.x, blockIdx.y, blockIdx.z,
blockDim.x, blockDim.y, blockDim.z,
gridDim.x, gridDim.y, gridDim.z);

}