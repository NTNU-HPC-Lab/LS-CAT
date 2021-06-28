#include "includes.h"

__global__ void cuda_sgemm(float* matrix_a, float* matrix_b, float* matrix_c, size_t M, size_t K, size_t N) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float sum = 0;
if (col < N && row < M) {
for (int k = 0; k < K; k++) {
sum +=
matrix_a[INDEX(row, k, M, K)] * matrix_b[INDEX(k, col, K, N)];
}
matrix_c[INDEX(row, col, M, N)] = sum;
}
}