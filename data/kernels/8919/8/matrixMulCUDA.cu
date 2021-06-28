#include "includes.h"
__global__ void matrixMulCUDA(float *C, float *A, float *B, int n)
{
int k;

// Get the row and the column in which thread resides in a block
int row = threadIdx.x;
int col = threadIdx.y;
float sum = 0.0f;
if (row >= n || col >= n) {
return;
}
for (k = 0; k < n; k++) {
sum += A[row * n + k] * B[k * n + col];

}
C[row * n + col] = sum;
}