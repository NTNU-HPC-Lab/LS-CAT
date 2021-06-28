#include "includes.h"
__global__ void init(int n, float *x, float *y) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride) {
x[i] = 1.0f;
y[i] = 2.0f;
}
}