#include "includes.h"
__global__ void matrixMulCUDA2(float *C, float *A, float *B, int n)
{
/*
Each thread computes more than 1 matrix elements
*/

// Define the starting row and ending row for each thread
int startRow = threadIdx.y * TILE_WIDTH;
int endRow = startRow + TILE_WIDTH;

// Define the starting column and ending column for each thread
int startCol = threadIdx.x * TILE_WIDTH;
int endCol = startCol + TILE_WIDTH;

// Now we have some block in 2 dimensions
for (int row = startRow; row < endRow; row++) {
for (int col = startCol; col < endCol; col++) {

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