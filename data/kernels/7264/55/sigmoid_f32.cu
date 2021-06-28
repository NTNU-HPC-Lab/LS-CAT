#include "includes.h"
__global__ void sigmoid_f32 (float* vector, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
float tmp = vector[idx];   output[idx] = 0.5 - 0.5 * tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
}
}