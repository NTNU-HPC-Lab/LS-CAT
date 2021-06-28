#include "includes.h"
__global__ void binarize_f32 (float* vector, float threshold, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
output[idx] = vector[idx] > threshold ? 1 : 0;
}
}