#include "includes.h"
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
// Calculate the row index of the Pd element and M
int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
// Calculate the column idenx of Pd and N
int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
float Pvalue = 0;
// each thread computes one element of the block sub-matrix
for (int k = 0; k < Width; ++k)
Pvalue += Md[Row*Width+k] * Nd[k*Width+Col];
Pd[Row*Width+Col] = Pvalue;
}