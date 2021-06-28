#include "includes.h"
__global__ void SolveSmoothGaussianGlobalKernel3(float* u, float* v, float* bku, float* bkv, int width, int height, int stride, float *outputu, float *outputv, float *outputbku, float* outputbkv)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float w[9] = {0.0f, 0.1667f, 0.0f, 0.1667f, 0.3333f, 0.1667f, 0.0f, 0.1667f, 0.0f};

float sumu = 0;
float sumv = 0;
for (int j = 0; j < 3; j++) {
for (int i = 0; i < 3; i++) {
//get values
int col = (ix + i - 1);
int row = (iy + j - 1);
if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
sumu = sumu + w[j * 3 + i] * u[col + stride*row];
sumv = sumv + w[j * 3 + i] * v[col + stride*row];
}
//solve gaussian
}
}
outputu[pos] = sumu;
outputv[pos] = sumv;
outputbku[pos] = bku[pos] + u[pos] - sumu;
outputbkv[pos] = bkv[pos] + v[pos] - sumv;
}