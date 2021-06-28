#include "includes.h"
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
//@@ Insert code to implement matrix multiplication here
//@@ You have to use shared memory for this MP
__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
int tx = threadIdx.x;
int ty = threadIdx.y;
int m = numARows;
int n = numBRows;
int k = numBColumns;

int numRows = blockIdx.y * blockDim.y + ty;
int numColumns = blockIdx.x * blockDim.x + tx;
float Cval = 0.0;

//Loading A and B elements and doing Boundary Check
for(int t = 0; t < (n-1)/TILE_WIDTH + 1; t++) {

if((numRows < numARows) && (t*TILE_WIDTH+tx < n)) {
ds_A[ty][tx] = A[numRows*n + t*TILE_WIDTH+tx];
} else {
ds_A[ty][tx] = 0.0;
}

if((numColumns < k) && (t*TILE_WIDTH+ty < n)) {
ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + numColumns];
} else {
ds_B[ty][tx] = 0.0;
}
__syncthreads();

for(int i = 0; i < TILE_WIDTH; i++) {
Cval += ds_A[ty][i] * ds_B[i][tx];
}
__syncthreads();
}

if(numRows < m && numColumns < k) {
C[numRows*k + numColumns] = Cval;
}
}