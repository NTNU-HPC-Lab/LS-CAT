#include "includes.h"
__global__ void sigmoidDeriv_f32 (float* vector, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
float tmp = 1.0 + (vector[idx] < 0.0 ? -vector[idx] : vector[idx]);   output[idx] = - 0.5 / (tmp*tmp);
}
}