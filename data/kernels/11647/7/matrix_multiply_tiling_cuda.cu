#include "includes.h"
__global__ void matrix_multiply_tiling_cuda(int* A, int* B, int* C, int m, int n) {
// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

// Index of the first sub-matrix of A processed by the block
int aBegin = n * blockDim.y * by;

// Index of the last sub-matrix of A processed by the block
int aEnd   = aBegin + n - 1;

// Step size used to iterate through the sub-matrices of A
int aStep  = blockDim.x;

// Index of the first sub-matrix of B processed by the block
int bBegin = blockDim.x * bx;

// Step size used to iterate through the sub-matrices of B
int bStep  = blockDim.y * m;

// Csub is used to store the element of the block sub-matrix
// that is computed by the thread
int Csub = 0;

// Loop over all the sub-matrices of A and B
// required to compute the block sub-matrix
for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
{

// Declaration of the shared memory array As used to
// store the sub-matrix of A
// Suppose to be As[blockDim.y][blockDim.x] but need dynamic allocation
// For simplicity, use a macro here
__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];

// Declaration of the shared memory array Bs used to
// store the sub-matrix of B
// Suppose to be Bs[blockDim.x][blockDim.y] but need dynamic allocation
// For simplicity, use a macro here
__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

// Load the matrices from device memory
// to shared memory; each thread loads
// one element of each matrix
As[ty][tx] = A[a + n * ty + tx];
Bs[ty][tx] = B[b + m * ty + tx];

// Synchronize to make sure the matrices are loaded
__syncthreads();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll

for (int k = 0; k < blockDim.x; ++k)
{
Csub += As[ty][k] * Bs[k][tx];
}

// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
__syncthreads();
}

// Write the block sub-matrix to device memory;
// each thread writes one element
int c = m * blockDim.y * by + blockDim.x * bx;
C[c + m * ty + tx] = Csub;
}