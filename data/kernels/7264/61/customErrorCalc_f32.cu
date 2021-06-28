#include "includes.h"
__global__ void customErrorCalc_f32 (float* vector, float* ideal_vector, float threshold, float scaleFoff, float scaleFon, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
float vectorValue = vector[idx];
if (ideal_vector[idx] > threshold) {
output[idx] = 1.0 - vectorValue;
if (vectorValue < threshold) {
output[idx] *= scaleFoff;
}
} else {
output[idx] = vectorValue * vectorValue;
if (vectorValue > threshold) {
output[idx] *= scaleFon;
}
}
}
}