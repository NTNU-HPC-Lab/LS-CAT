#include "includes.h"
__global__ void pw_tanh(float *y, float *a, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] = tanh(a[i]);
}