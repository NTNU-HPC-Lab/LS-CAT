#include "includes.h"
__global__ void warmup(float *input, float *output) {

const int i = threadIdx.x + blockIdx.x * blockDim.x;
output[i] = input[i] * input[i];
}