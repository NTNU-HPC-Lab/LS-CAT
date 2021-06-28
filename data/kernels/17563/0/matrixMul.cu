#include "includes.h"
__global__ void matrixMul(int* A, int* B, int* C, int aF, int aC, int bF, int bC, int cF, int cC) {
// Compute each thread's global row and column index
int row = (blockIdx.y * blockDim.y) + threadIdx.y;
int col = (blockIdx.x * blockDim.x) + threadIdx.x;

// Iterate over row, and down column
////c[row * N + col] = 0;
if (aC != bF) return;
if ((row < aF) && (col < bC)) {
for (int k = 0; k < aC; ++k) {
// Accumulate results for a single element
C[row * cC + col] += A[row * aC + k] * B[k * bC + col];
}
}
//C[row * aF + col] = 0;
}