#include "includes.h"
__global__ void nmfcpy(double *mat, double *matcp, int m, int n) //kernel copy must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

if (row < m && col < n)
mat[row*n+col] = matcp[row*n+col];
}