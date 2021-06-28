#include "includes.h"
__global__ void SolveSmoothGaussianGlobalKernel5(float* u, float* v, float* bku, float* bkv, int width, int height, int stride, float *outputu, float *outputv, float *outputbku, float* outputbkv)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float w[25] = { 0.0037, 0.0147, 0.0256, 0.0147, 0.0037,
0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
0.0256, 0.0952, 0.1502, 0.0952, 0.0256,
0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
0.0037, 0.0147, 0.0256, 0.0147, 0.0037 };

float sumu = 0;
float sumv = 0;
for (int j = 0; j < 5; j++) {
for (int i = 0; i < 5; i++) {
//get values
int col = (ix + i - 2);
int row = (iy + j - 2);
if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
sumu = sumu + w[j * 5 + i] * u[col + stride*row];
sumv = sumv + w[j * 5 + i] * v[col + stride*row];
}
//solve gaussian
}
}
outputu[pos] = sumu;
outputv[pos] = sumv;
outputbku[pos] = bku[pos] + u[pos] - sumu;
outputbkv[pos] = bkv[pos] + v[pos] - sumv;
}