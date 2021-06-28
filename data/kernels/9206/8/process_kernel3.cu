#include "includes.h"
__global__ void process_kernel3(const float* input, float* output, int numElements){

int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int globalThreadId = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;

if(globalThreadId < numElements)
output[globalThreadId] = (float)sqrt(input[globalThreadId]);
}