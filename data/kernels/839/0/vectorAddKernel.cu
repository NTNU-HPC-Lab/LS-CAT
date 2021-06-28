#include "includes.h"

__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceResult) {
unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
// insert operation here
deviceResult[i] = deviceA[i]+deviceB[i];
}