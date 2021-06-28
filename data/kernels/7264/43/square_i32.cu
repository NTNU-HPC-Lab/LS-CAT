#include "includes.h"
__global__ void square_i32 (int* vector, int* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
output[idx] = vector[idx] * vector[idx];
}
}