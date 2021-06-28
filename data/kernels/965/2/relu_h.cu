#include "includes.h"
__global__ void relu_h(float *X, float *Y, int size_in) {
int t = blockIdx.x * blockDim.x + threadIdx.x;
if (t < size_in) {
Y[t] = 0.0;
if (X[t] >= 0)
Y[t] = X[t];
}
}