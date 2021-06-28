#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void matrixMultiplicationKernelNaive(const float* A, const float* B, float* C, int a, int b, int c, int d) {

int ROW = blockIdx.y*blockDim.y+threadIdx.y;
int COL = blockIdx.x*blockDim.x+threadIdx.x;

float tmpSum = 0.0f;

if (ROW < a && COL < d) {
// each thread computes one element of the block sub-matrix
for (int ii = 0; ii < b; ii++) {
tmpSum += A[ROW * b + ii] * B[ii * b + COL];
}
}
C[ROW * a + COL] = tmpSum;
}