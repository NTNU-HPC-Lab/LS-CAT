#include "includes.h"
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
//@@ Insert code to implement matrix multiplication here
int iRow = blockIdx.y*blockDim.y+threadIdx.y;
int iCol = blockIdx.x*blockDim.x+threadIdx.x;
if(( iRow < numARows) && (iCol < numBColumns)) {
float Cvalue = 0.0;
for (int i = 0;i< numAColumns;++i)
{
Cvalue += A[iRow*numAColumns+i]*B[iCol+i*numBColumns];
}
C[iRow*numBColumns+iCol] = Cvalue;
}
}