#include "includes.h"
__global__ void matrixMulCUDA5(float *C, float *A, float *B, unsigned int n)
{

const int tileWidth = 1;

// Define the starting row and ending row for each thread block
int startRow = blockIdx.y * blockDim.y + threadIdx.y * tileWidth;
int endRow = startRow + tileWidth;

// Define the starting column and ending column for each thread block
int startCol = blockIdx.x * blockDim.x + threadIdx.x * tileWidth;
int endCol = startCol + tileWidth;

// Each block of threads allocate space on shared memory
__shared__ float A_S[32 * 32 * 4];
__shared__ float B_S[32 * 32 * 4];

// Each thread helps copying the proper indexes into the shared memory
// Now we have some blocks in 2 dimensions
for (int row = startRow; row < endRow; row++) {
for (int col = startCol; col < endCol; col++) {



// Copy data into shared memory
for (int k = 0; k < n; k++) {
A_S[k] = A[row * n + k];
B_S[k] = B[k * n + col];
}

// Synchronize all threads to make a tile completely ready to go!
__syncthreads();

// Compute the proper sum for each block
float sum = 0.0f;	// Defined as a register (Better than directly writing to C)
for (int k = 0; k < n; k++) {
sum += A_S[k] * B_S[k];
}

// Write back sum into C
C[row * n + col] = sum;
}
}
}