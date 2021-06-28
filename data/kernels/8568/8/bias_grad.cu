#include "includes.h"
__global__ void bias_grad(float *pre_grad, float *output, int rows, int cols) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= rows) return;
output[i] = 0;
for (int k = 0; k < cols; k++) {
output[i] += pre_grad[i * cols + k];
}
}