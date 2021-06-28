#include "includes.h"
__global__ void mult2Matrix(float *M, float *N, float *P) {
// Calculate the row index of the P element and M
int Row = blockIdx.y * blockDim.y + threadIdx.y;
// Calculate the column index of P and N
int Col = blockIdx.x * blockDim.x + threadIdx.x;
if ((Row < WIDTH) && (Col < WIDTH)) {
float Pvalue = 0;
// each thread computes one element of the block sub-matrix
for (int k = 0; k < WIDTH; ++k) {
Pvalue += M[Row*WIDTH + k] * N[k*WIDTH + Col];
}
P[Row*WIDTH + Col] = Pvalue;
}
}