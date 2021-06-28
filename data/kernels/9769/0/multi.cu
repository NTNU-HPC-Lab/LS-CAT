#include "includes.h"
__global__ void multi(float *a, float *b, float *c, int width) {
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

float result = 0;

if (col < width && row < width) {
for (int k = 0; k < width; k++) {
result += a[row * width + k] * b[k * width + col];
}
c[row * width + col] = result;
}
}