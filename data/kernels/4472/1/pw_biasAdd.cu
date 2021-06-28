#include "includes.h"
__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] += bias[i % nBias];
}