#include "includes.h"
__global__ void reluDeriv_f32 (float* vector, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
output[idx] = vector[idx] > 0.0 ? 1.0 : 0.0;
}
}