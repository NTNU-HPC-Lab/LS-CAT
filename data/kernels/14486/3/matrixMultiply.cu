#include "includes.h"
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {

// which thread is this?
int tx = blockIdx.x*blockDim.x + threadIdx.x;
int ty = blockIdx.y*blockDim.y + threadIdx.y;

if ((tx < numCRows) && (ty < numCColumns)){
float value = 0.0f;
for (unsigned int i = 0; i < numAColumns; ++i){
value += A[tx*numAColumns + i] * B[i*numBColumns + ty];
}
C[tx*numCColumns + ty] = value;
}

}