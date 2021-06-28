#include "includes.h"
__global__ void tanhDeriv_f32 (float* vector, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
float tmp = vector[idx] < 0.0 ? -vector[idx] : vector[idx];   output[idx] =  1.0 / ((1.0+tmp)*(1.0+tmp));
}
}