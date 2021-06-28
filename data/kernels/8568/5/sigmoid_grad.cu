#include "includes.h"
__global__ void sigmoid_grad(float *pre_grad, float *output, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
float t = output[i * cols + j];
pre_grad[i * cols + j] *= t * (1 - t);
}