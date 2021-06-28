#include "includes.h"
__global__ void matrixMultiply1(float *A, float *C, int size) {
int Col = blockDim.y * blockIdx.y + threadIdx.y;
int Row = blockDim.x * blockIdx.x + threadIdx.x;


for(int k = 0; k < size; k++)
C[Row * size + Col] += A[k * size + Row] * A[k * size + Col];

}