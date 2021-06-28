#include "includes.h"
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
int Row = blockIdx.y*blockDim.y+threadIdx.y;// Calculate the row index of the P element and M
int Col = blockIdx.x*blockDim.x+threadIdx.x;// Calculate the column index of P and N
if ((Row < Width) && (Col < Width)) {
float Pvalue = 0;
for (int k = 0; k < Width; ++k) {
Pvalue += M[Row*Width+k]*N[k*Width+Col];// each thread computes one element of the block sub-matrix
}

P[Row*Width+Col] = Pvalue;
}
}