#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void matrixMultiplicationKernelEW(const float* A, const float* B, float* C, int a, int b) {

int ROW = blockIdx.y*blockDim.y+threadIdx.y;
int COL = blockIdx.x*blockDim.x+threadIdx.x;

if (ROW < a && COL < b) {
C[ROW * a + COL] = A[ROW * b + COL]*B[ROW * b + COL];
}
}