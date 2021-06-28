#include "includes.h"

#define TILE_WIDTH 16

// Compute C = A * B



__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {


//@@ Insert code to implement matrix multiplication here
int r = blockIdx.x * blockDim.x + threadIdx.x;
int c = blockIdx.y * blockDim.y + threadIdx.y;

if ((r < numCRows) && (c < numCColumns)){
float value = 0.0;

for (int i=0; i < numAColumns; i++){
value += A[r*numAColumns+i] * B[i*numBColumns+c];
}
C[r*numCColumns+c] = value;
}

}