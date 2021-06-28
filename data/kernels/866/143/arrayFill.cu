#include "includes.h"
__global__ void arrayFill(float* data, float value, int size) {
int stride = gridDim.x * blockDim.x;
int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < size; i += stride) data[i] = value;
}