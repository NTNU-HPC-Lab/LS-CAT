#include "includes.h"
__global__ void matrixMultiplyShared(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int r = by * blockDim.y + ty;
int c = bx * blockDim.x + tx;
int dimC = numAColumns;

__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

float value = 0.0;

for (int i=0; i < (dimC-1)/TILE_WIDTH +1; i++){


if ((r < numCRows) && ((i*TILE_WIDTH + tx)< dimC)){
ds_A[ty][tx]=A[r*dimC + i*TILE_WIDTH + tx];
} else {
ds_A[ty][tx]=0.0;
}

if ((c < numCColumns) && ((i*TILE_WIDTH + ty)< dimC)){
ds_B[ty][tx]=B[(i*TILE_WIDTH + ty)*numBColumns + c];
} else {
ds_B[ty][tx]=0.0;
}

__syncthreads();

for (int j=0; j<TILE_WIDTH; j++){
value += ds_A[ty][j] * ds_B[j][tx];
}

__syncthreads();

}

if ((r < numCRows) && (c < numCColumns)){
C[r*numCColumns+c] = value;
}
}