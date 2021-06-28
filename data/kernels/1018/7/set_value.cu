#include "includes.h"
__global__ void set_value(float value, float *array, unsigned int size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
array[index] = value;
}