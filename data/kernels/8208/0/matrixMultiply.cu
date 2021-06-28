#include "includes.h"


#define TILE_WIDTH = 16;
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {

//@@Y-axis matrix dimension
int row = blockIdx.y*blockDim.y + threadIdx.y;

//@@X-axis matrix Dimension
int columns = blockIdx.x*blockDim.x + threadIdx.x;

//@@Initilaizing final value to add in the output matrix
float pValue = 0;

//@@checking for boundary condition
if (row < numARows  && columns < numBColumns) {

//@@adding values from 0 to matrix A width and from 0 to martrix B Height
for (int k = 0; k < numAColumns; k++) {

//@@Summation of the rows from matrix A and columns from matrix B to pValue
pValue += A[row*numAColumns + k] * B[k*numBColumns + columns];

}

//@@add the final value to the output matrix
C[row*numCColumns + columns] = pValue;

}

}