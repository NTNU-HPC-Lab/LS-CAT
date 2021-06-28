#include "includes.h"
__global__ void convKernel(const float* source, const float* kernel, float* target, const int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx >= len) return;

float value = 0.0f;

for (int i = 0; i < len; i++) {
value += source[i] *
kernel[(len + len / 2 + idx - i) % len]; // Positive modulo
}

target[idx] = value;
}