#include "includes.h"



__global__ void matrixAddKernel(float* A, float* B, float* C, int n)
{
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

if((Row < n) && (Col < n))
C[Row * n + Col] = A[Row * n + Col] + B[Row * n + Col];
}