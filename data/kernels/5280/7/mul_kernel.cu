#include "includes.h"
__global__ void mul_kernel(const int n, const float *a, const float *b, float *y) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);
i += blockDim.x * gridDim.x) {
y[i] = a[i] * b[i];
}
}