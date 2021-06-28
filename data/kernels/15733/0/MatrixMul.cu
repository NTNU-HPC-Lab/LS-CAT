#include "includes.h"

// CUDA kernel for vector addition

// Initialize
__global__ void MatrixMul(int* a, int* b, int* c, int n) {
// row
int row = (blockIdx.y * blockDim.y) + threadIdx.y;
//col
int col = (blockIdx.x * blockDim.x) + threadIdx.x;
int temp_sum = 0;
// boundary guard
if ((row < n) && (col < n)) {
for (int k = 0; k < n; k++)
{
temp_sum += a[row*n+k]*b[k*n+col];
}
c[row*n+col] = temp_sum;
}
}