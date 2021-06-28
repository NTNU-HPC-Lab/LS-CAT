#include "includes.h"
__global__ void backward_relu_h(float *X, float *Y, int size_in) {
int t = blockIdx.x * blockDim.x + threadIdx.x;
if (t < size_in) {
X[t] = 0.0;
if (X[t] >= 0)
X[t] = Y[t];
}
}