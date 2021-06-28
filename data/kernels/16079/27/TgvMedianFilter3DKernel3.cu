#include "includes.h"
__global__ void TgvMedianFilter3DKernel3(float* X, float* Y, float *Z, int width, int height, int stride, float *X1, float *Y1, float *Z1)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float mX[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
float mY[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
float mZ[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

for (int j = 0; j < 3; j++) {
for (int i = 0; i < 3; i++) {
//get values
int col = (ix + i - 1);
int row = (iy + j - 1);
int index = j * 3 + i;
if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
mX[index] = X[col + stride * row];
mY[index] = Y[col + stride * row];
mZ[index] = Z[col + stride * row];
}
else if ((col < 0) && (row >= 0) && (row < height)) {
mX[index] = X[stride*row];
mY[index] = Y[stride*row];
mZ[index] = Z[stride*row];
}
else if ((col > width) && (row >= 0) && (row < height)) {
mX[index] = X[width - 1 + stride * row];
mY[index] = Y[width - 1 + stride * row];
mZ[index] = Z[width - 1 + stride * row];
}
else if ((col >= 0) && (col < width) && (row < 0)) {
mX[index] = X[col];
mY[index] = Y[col];
mZ[index] = Z[col];
}
else if ((col >= 0) && (col < width) && (row > height)) {
mX[index] = X[col + stride * (height - 1)];
mY[index] = Y[col + stride * (height - 1)];
mZ[index] = Z[col + stride * (height - 1)];
}
//solve gaussian
}
}

float tmpX, tmpY, tmpZ;
for (int j = 0; j < 5; j++) {
for (int i = j + 1; i < 9; i++) {
if (mX[j] > mX[i]) {
//Swap the variables.
tmpX = mX[j];
mX[j] = mX[i];
mX[i] = tmpX;
}
if (mY[j] > mY[i]) {
//Swap the variables.
tmpY = mY[j];
mY[j] = mY[i];
mY[i] = tmpY;
}
if (mZ[j] > mZ[i]) {
//Swap the variables.
tmpZ = mZ[j];
mZ[j] = mZ[i];
mZ[i] = tmpZ;
}
}
}

X1[pos] = mX[4];
Y1[pos] = mY[4];
Z1[pos] = mZ[4];
}