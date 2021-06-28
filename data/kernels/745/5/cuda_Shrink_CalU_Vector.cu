#include "includes.h"
__global__ void cuda_Shrink_CalU_Vector(float *Y, float *U, float *X, float lambda, float *L1Weight, int nRows, int nCols, int nFilts) {
unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

float WLambda;
float absxV1, X_temp, U_temp, Y_temp;

if ((Tidx < nCols) && (Tidy < nRows)) {

for (int k = 0; k < nFilts; k += 1) {
index = Tidx + (Tidy + nRows * k) * nCols;

X_temp = (X[index] / (nRows * nCols));
U_temp = U[index];

WLambda = lambda * L1Weight[k];

Y_temp = X_temp + U_temp;
absxV1 = fabs(Y_temp) - WLambda;

Y_temp = signbit(-absxV1) * copysign(absxV1, Y_temp);

Y[index] = Y_temp;
U[index] = U_temp + X_temp - Y_temp;
}
}
}