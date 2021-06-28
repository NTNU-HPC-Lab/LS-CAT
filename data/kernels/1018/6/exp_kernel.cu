#include "includes.h"
__global__ void exp_kernel(float *array, unsigned int size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
array[index] = exp(array[index]);
}