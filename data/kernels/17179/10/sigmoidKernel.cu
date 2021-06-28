#include "includes.h"
__global__ void sigmoidKernel(float* input, float* output, int edge) {

KERNEL_POSITION;
output[position] = 1 / (1 + exp(-input[position]));
}