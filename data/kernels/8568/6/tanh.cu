#include "includes.h"
__global__ void tanh(float *inout, float *bias, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
inout[i * cols + j] = tanhf(inout[i * cols + j]) + bias[i];
}