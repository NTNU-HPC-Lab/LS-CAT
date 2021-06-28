#include "includes.h"
__global__ void sigmoid(float *inout, float *bias, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
float t = inout[i * cols + j];
inout[i * cols + j] = 1 / (1 + expf(-t)) + bias[i];
}