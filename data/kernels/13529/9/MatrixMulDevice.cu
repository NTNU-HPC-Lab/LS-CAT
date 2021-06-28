#include "includes.h"
__global__ void MatrixMulDevice( float *A, float *B, float *C, int *matrixSize)
{
int chunk = (*matrixSize) / gridDim.x;
int sum, i, k;

for(i = blockIdx.x * chunk; i < blockIdx.x * chunk + chunk - 1; i++) {
sum = 0;

for(k = 0; k < *matrixSize; k++) {
sum += A[i * *matrixSize + k] * B [k * *matrixSize + threadIdx.x];
}

C[i * *matrixSize + threadIdx.x] = sum;
}
}