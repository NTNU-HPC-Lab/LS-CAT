#include "includes.h"
__global__ void relu(float *inout, float *bias, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
inout[i * cols + j] = fmaxf(0.0, inout[i * cols + j] + bias[i]);
}