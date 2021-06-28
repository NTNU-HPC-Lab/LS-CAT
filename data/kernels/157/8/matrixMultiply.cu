#include "includes.h"
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
//@@ Insert code to implement matrix multiplication here
// Calculate the row index
int numRows = blockIdx.y*blockDim.y+threadIdx.y;
// Calculate the column index
int numColumns = blockIdx.x*blockDim.x+threadIdx.x;
if ((numRows < numARows) && (numColumns < numBColumns)) {
float Cval = 0.0;
// Each thread computes one element of the block sub-matrix
for (int k = 0; k < numBRows; ++k) {
Cval += A[numRows*numBRows+k]*B[numColumns+k*numBColumns];
}
C[numRows*numBColumns+numColumns] = Cval;
}
}