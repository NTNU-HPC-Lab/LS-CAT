#include "includes.h"
__global__ void simpleMPIKernel(float *input, float *output) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
output[tid] = sqrt(input[tid]);
}