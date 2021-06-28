#include "includes.h"
__global__ void MatrixMul(float *A, float *B, float *C, int n)
{
// Each thread computes a single element of C
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

float sum = 0;
for (int i = 0; i < n; ++i) {
sum += (A[row*n + i] * B[i*n + col]);
}

C[row*n + col] = sum;
printf("\n Block[%d][%d] : Thread[%d][%d] : Product = %.2f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, sum);
}