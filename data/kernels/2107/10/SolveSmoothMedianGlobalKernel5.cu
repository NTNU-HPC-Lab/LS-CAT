#include "includes.h"
__global__ void SolveSmoothMedianGlobalKernel5(float* u, float* v, float* bku, float* bkv, int width, int height, int stride, float *outputu, float *outputv, float *outputbku, float* outputbkv)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float mu[25] = { 0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0 };

float mv[25] = { 0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0 };

for (int j = 0; j < 5; j++) {
for (int i = 0; i < 5; i++) {
//get values
int col = (ix + i - 2);
int row = (iy + j - 2);
if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
mu[j * 5 + i] = u[col + stride*row];
mv[j * 5 + i] = v[col + stride*row];
}
else if ((col < 0) && (row >= 0) && (row < height)) {
mu[j * 5 + i] = u[stride*row];
mv[j * 5 + i] = v[stride*row];
}
else if ((col > width) && (row >= 0) && (row < height)) {
mu[j * 5 + i] = u[width - 1 + stride*row];
mv[j * 5 + i] = v[width - 1 + stride*row];
}
else if ((col >= 0) && (col < width) && (row < 0)) {
mu[j * 5 + i] = u[col];
mv[j * 5 + i] = v[col];
}
else if ((col >= 0) && (col < width) && (row > height)) {
mu[j * 5 + i] = u[col + stride*(height - 1)];
mv[j * 5 + i] = v[col + stride*(height - 1)];
}
//solve gaussian
}
}

float tmpu, tmpv;
for (int j = 0; j < 25; j++) {
for (int i = j+1; i < 25; i++) {
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

outputu[pos] = mu[12];
outputv[pos] = mv[12];
outputbku[pos] = bku[pos] + u[pos] - mu[12];
outputbkv[pos] = bkv[pos] + v[pos] - mv[12];
}