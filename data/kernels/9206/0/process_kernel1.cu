#include "includes.h"

// Device Function for process_kernel1


// Device Function for process_kernel2


// Device Function for process_kernel3

__global__ void process_kernel1(const float* input1, const float* input2, float* output, int numElements){

int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int globalThreadId = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;

if(globalThreadId < numElements)
output[globalThreadId] = (float)sin(input1[globalThreadId]) + (float)cos(input2[globalThreadId]);

}