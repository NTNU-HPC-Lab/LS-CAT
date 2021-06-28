#include "includes.h"
__global__ void gpuMatMul(float *a, float *b, float *c, int m, int n, int p) {
uint i = blockDim.x * blockIdx.x + threadIdx.x;
uint j = blockDim.y * blockIdx.y + threadIdx.y;
if (i < m && j < p) {
float val = 0;
for (int k = 0; k < n; ++k) {
val += a[i * n + k] * b[k * p + j];
}
c[i * p + j] = val;
}
}