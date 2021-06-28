#include "includes.h"
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
int Row = blockIdx.y * blockDim.y + threadIdx.y;

int Col = blockIdx.x * blockDim.x + threadIdx.x;

if((Row < Width) && (Col < Width))
{
float Pvalue = 0;
for(int k = 0; k < Width; ++k)
{
Pvalue += M[Row*Width+k]*N[k*Width+Col];
}
P[Row*Width+Col] = Pvalue;
}
}