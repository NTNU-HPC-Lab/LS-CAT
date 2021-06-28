#include "includes.h"
__global__ void add(float *array_a, float *array_b, float *array_c, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int step = blockDim.x * gridDim.x;

for (int i = tid; i < size; i += step) {
array_c[i] = array_a[i] + array_b[i];
}
}