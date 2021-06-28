#include "includes.h"
__global__ void matrixSum(const double * M1,const double * M2,double * Msum,double alpha,double beta, int rows, int cols)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row < rows && col < cols){
Msum[row + col*rows] = alpha*M1[row+col*rows]+beta*M2[row+col*rows];
}
}