#include "includes.h"
__global__ void times(float *input, unsigned int input_size, float *output, unsigned int n) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < n * input_size)
output[index] = input[index % input_size];
}