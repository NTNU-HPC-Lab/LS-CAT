#include "includes.h"
__global__ void matrixMulCUDA4(float *C, float *A, float *B, unsigned int n)
{
/*
Each block computes a tile
*/
int tileWidth = 32;

// Define the starting row and ending row for each thread
int startRow = blockIdx.y * blockDim.y + threadIdx.y * tileWidth;
int endRow = startRow + tileWidth;

// Define the starting column and ending column for each thread
int startCol = blockIdx.x * blockDim.x + threadIdx.x * tileWidth;
int endCol = startCol + tileWidth;

// Now we have some block in 2 dimensions
for (int row = startRow; row < endRow; row++) {
for (int col = startCol; col < endCol; col++) {

if (row >= n || col >= n) {
continue;
}

// Compute the proper sum for each block
float sum = 0.0f;	// Defined as a register (Better than directly writing to C)
for (int k = 0; k < n; k++) {
sum += A[row * n + k] * B[k * n + col];
}

// Write back sum into C
C[row * n + col] = sum;
}
}
}