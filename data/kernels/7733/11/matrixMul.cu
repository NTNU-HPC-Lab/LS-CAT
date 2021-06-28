#include "includes.h"
__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if (row < rows && col < cols){
for (int k = 0; k < cols2; k++){
C[row*cols+col]+=b[k*cols+col]*a[row*cols2+k];
}
}
}