#include "includes.h"

#define BLOCK_SIZE 16



__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

if( Col < Width && Row < Width)
{
float Pvalue = 0;
for(int k = 0; k < Width; ++k)
{
Pvalue += M[Row * Width + k] * N[k * Width + Col];
}
P[Row * Width + Col] = Pvalue;
}
}