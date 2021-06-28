#include "includes.h"
__global__ void SolveSmoothMedianGlobalKernel3(float* u, float* v, float* bku, float* bkv, int width, int height, int stride, float *outputu, float *outputv, float *outputbku, float* outputbkv)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float mu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

float mv[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

for (int j = 0; j < 3; j++) {
for (int i = 0; i < 3; i++) {
//get values
int col = (ix + i - 1);
int row = (iy + j - 1);
int index = j * 3 + i;
if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
mu[index] = u[col + stride*row];
mv[index] = v[col + stride*row];
}
else if ((col < 0) && (row >= 0) && (row < height)) {
mu[index] = u[stride*row];
mv[index] = v[stride*row];
}
else if ((col > width) && (row >= 0) && (row < height)) {
mu[index] = u[width - 1 + stride*row];
mv[index] = v[width - 1 + stride*row];
}
else if ((col >= 0) && (col < width) && (row < 0)) {
mu[index] = u[col];
mv[index] = v[col];
}
else if ((col >= 0) && (col < width) && (row > height)) {
mu[index] = u[col + stride*(height - 1)];
mv[index] = v[col + stride*(height - 1)];
}
//solve gaussian
}
}

float tmpu, tmpv;
for (int j = 0; j < 9; j++) {
for (int i = j + 1; i < 9; i++) {
if (mu[j] > mu[i]) {
//Swap the variables.
tmpu = mu[j];
mu[j] = mu[i];
mu[i] = tmpu;
}
if (mv[j] > mv[i]) {
//Swap the variables.
tmpv = mv[j];
mv[j] = mv[i];
mv[i] = tmpv;
}
}
}

outputu[pos] = mu[4];
outputv[pos] = mv[4];
outputbku[pos] = bku[pos] + u[pos] - mu[4];
outputbkv[pos] = bkv[pos] + v[pos] - mv[4];
}