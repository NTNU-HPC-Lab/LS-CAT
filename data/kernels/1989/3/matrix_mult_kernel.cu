#include "includes.h"
__global__ void matrix_mult_kernel(int *a, int *b, int *c, int m, int n, int k) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int sum = 0;
if (col < n && row < m) {
for (int i = 0; i < k; i++) {
sum += a[row * k + i] * b[i * n + col];
}
c[row * n + col] = sum;
}
}