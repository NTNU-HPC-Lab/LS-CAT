#include "includes.h"
__global__ void matrixMultiply3(float* A, float* C, int size) {

float CValue = 0;

int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

__shared__ float As[TILE_WIDTH][TILE_WIDTH];

for (int k = 0; k < (TILE_WIDTH + size - 1)/TILE_WIDTH; k++) {

if (k * TILE_WIDTH + threadIdx.x < size && Row < size)
As[threadIdx.y][threadIdx.x] = A[Row * size + k * TILE_WIDTH + threadIdx.x];
else
As[threadIdx.y][threadIdx.x] = 0.0;

if (k * TILE_WIDTH + threadIdx.y < size && Col < size)
As[threadIdx.y][threadIdx.x] = A[(k*TILE_WIDTH + threadIdx.y) * size + Col];
else
As[threadIdx.y][threadIdx.x] = 0.0;

__syncthreads();

for (int n = 0; n < TILE_WIDTH; ++n)
CValue += As[threadIdx.y][n] * As[n][threadIdx.x];

__syncthreads();
}

if (Row < size && Col < size)
C[((blockIdx.y * blockDim.y + threadIdx.y) * size) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;
}