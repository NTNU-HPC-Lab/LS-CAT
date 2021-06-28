#include "includes.h"
__global__ void tanh_grad(float *pre_grad, float *output, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
float t = output[i * cols + j];
pre_grad[i * cols + j] *= 1 - t * t;
}