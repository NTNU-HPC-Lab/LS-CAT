#include "includes.h"
__global__ void matrixTrans(double * M,double * MT, int rows, int cols)
{
double val=0;
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if (row < rows && col < cols){
val = M[col + row*cols];
MT[row + col*rows] = val;
}
}