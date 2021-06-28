#include "includes.h"
__global__ void matrixMultiply(float *A,float *B,float *C,int numARows,int numAColumns,int numBRows,int numBColumns,int numCRows,int numCColumns)
{
// variable declarations
int row=blockIdx.y * blockDim.y + threadIdx.y;
int col=blockIdx.x * blockDim.x + threadIdx.x;
// code
if((row < numARows) && (col < numBColumns))
{
float Cvalue=0.0;
for(int k=0; k < numAColumns; k++)
{
Cvalue +=A[row * numAColumns + k] * B[k * numBColumns + col];
}
C[row * numCColumns + col]=Cvalue;
}
}