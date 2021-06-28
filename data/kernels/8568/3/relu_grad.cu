#include "includes.h"
__global__ void relu_grad(float *pre_grad, float *output, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
if (output[i * cols + j] <= 0)
pre_grad[i * cols + j] = 0;
}