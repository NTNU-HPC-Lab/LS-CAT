#include "includes.h"
__global__ void cuda_Cal_X_minus_U_W(float *Y, float *U, float *X, int *Weight, int nRows, int nCols) {
unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

float X_temp, U_temp, Y_temp;

if ((Tidx < nCols) && (Tidy < nRows)) {
index = Tidx + Tidy * nCols;

X_temp = (X[index] / (nRows * nCols));
U_temp = U[index];

Y_temp = (1 - Weight[index]) * (X_temp + U_temp);

Y[index] = Y_temp;
U[index] = U_temp + X_temp - Y_temp;
}
}