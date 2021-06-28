#include "includes.h"
__global__ void MatrixMulVarKernel(float* M, float* N, float* P, int widthAHeightB, int heightA, int widthB) {
int Row = blockIdx.y*blockDim.y+threadIdx.y;// Calculate the row index of the P element and M
int Col = blockIdx.x*blockDim.x+threadIdx.x;// Calculate the column index of P and N
if ((Row < heightA) && (Col < widthB)) {
float Pvalue = 0;
for (int k = 0; k < widthAHeightB; ++k) {
Pvalue += M[Row*widthAHeightB+k]*N[k*widthB+Col];// each thread computes one element of the block sub-matrix
}

P[Row*widthB+Col] = Pvalue;
}
}