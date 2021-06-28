#include "includes.h"
__global__ void attentionKernel(float *x, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;

if (j >= cols) return;
float sum = 0;
for (int k = 0; k < rows; k++) {
sum += x[k * cols + j];
}
for (int k = 0; k < rows; k++) {
x[k * cols + j] *= sum;
}
}